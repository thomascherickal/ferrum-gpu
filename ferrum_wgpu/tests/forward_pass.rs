#![forbid(unsafe_code)]
//! Whole GPT-2 forward pass (embed, two blocks, final norm, head, loss) on the GPU against the
//! same pass composed from the CPU reference ops. Skips without an adapter.

use ferrum_ref::matmul::{matmul, rand_vec};
use ferrum_ref::nn;
use ferrum_wgpu::device::open_for_test;
use ferrum_wgpu::matmul::{matmul_host, Variant};
use ferrum_wgpu::ops;
use ferrum_wgpu::tensor::GpuBuf;
use ferrum_wgpu::tolerance::{max_rel_err, FORWARD};
use ferrum_wgpu::Gpu;

const BATCH: usize = 2;
const T: usize = 6;
const C: usize = 8;
const HEADS: usize = 2;
const HIDDEN: usize = 20;
const VOCAB: usize = 13;
const BLOCKS: usize = 2;
const ROWS: usize = BATCH * T;

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    k: usize,
    n: usize,
}

fn lin(k: usize, n: usize, seed: u64) -> Linear {
    let w = rand_vec(k * n, seed).into_iter().map(|v| v * 0.5).collect();
    Linear {
        w,
        b: rand_vec(n, seed + 1),
        k,
        n,
    }
}

struct Block {
    ln1: (Vec<f32>, Vec<f32>),
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: (Vec<f32>, Vec<f32>),
    f1: Linear,
    f2: Linear,
}

fn block(seed: u64) -> Block {
    let ln = |s| {
        (
            rand_vec(C, s).into_iter().map(|v| 1.0 + 0.1 * v).collect(),
            rand_vec(C, s + 1),
        )
    };
    Block {
        ln1: ln(seed),
        q: lin(C, C, seed + 10),
        k: lin(C, C, seed + 20),
        v: lin(C, C, seed + 30),
        o: lin(C, C, seed + 40),
        ln2: ln(seed + 50),
        f1: lin(C, HIDDEN, seed + 60),
        f2: lin(HIDDEN, C, seed + 70),
    }
}

fn cpu_linear(x: &[f32], l: &Linear, rows: usize) -> Vec<f32> {
    nn::add_bias(&matmul(x, &l.w, rows, l.k, l.n).unwrap(), &l.b)
}

fn gpu_linear(g: &Gpu, x: &GpuBuf, l: &Linear, rows: usize, host_x: &[f32]) -> GpuBuf {
    // ponytail: matmul goes through the host-to-host helper here; the kernel itself is the
    // Phase 0 one. A resident-buffer composition arrives with the model engine in Phase 3.
    let _ = x;
    let y = matmul_host(g, Variant::Blocked4, host_x, &l.w, (rows, l.k, l.n)).unwrap();
    let (gy, gb, out) = (
        g.upload(&y).unwrap(),
        g.upload(&l.b).unwrap(),
        g.zeros(rows * l.n).unwrap(),
    );
    ops::add_bias(g, &gy, &gb, &out).unwrap();
    out
}

#[test]
fn full_forward_pass_matches_the_cpu_reference() {
    let Some(g) = open_for_test() else { return };
    let tok = rand_vec(VOCAB * C, 1);
    let pos = rand_vec(T * C, 2);
    let blocks: Vec<Block> = (0..BLOCKS as u64).map(|i| block(100 * (i + 1))).collect();
    let lnf: (Vec<f32>, Vec<f32>) = (
        rand_vec(C, 3).into_iter().map(|v| 1.0 + 0.1 * v).collect(),
        rand_vec(C, 4),
    );
    let head = lin(C, VOCAB, 5);
    let ids: Vec<usize> = (0..ROWS).map(|i| (i * 5 + 2) % VOCAB).collect();
    let targets: Vec<usize> = (0..ROWS).map(|i| (i * 3 + 1) % VOCAB).collect();
    let hd = C / HEADS;

    // CPU reference.
    let mut x = nn::embed(&tok, &pos, &ids, T, C);
    for b in &blocks {
        let h = nn::layernorm(&x, &b.ln1.0, &b.ln1.1, ROWS, C);
        let (q, k, v) = (
            cpu_linear(&h, &b.q, ROWS),
            cpu_linear(&h, &b.k, ROWS),
            cpu_linear(&h, &b.v, ROWS),
        );
        let (a, _) = nn::attention(&q, &k, &v, (BATCH, T, HEADS, hd));
        x = nn::add(&x, &cpu_linear(&a, &b.o, ROWS));
        let h = nn::layernorm(&x, &b.ln2.0, &b.ln2.1, ROWS, C);
        let f: Vec<f32> = cpu_linear(&h, &b.f1, ROWS)
            .into_iter()
            .map(nn::gelu)
            .collect();
        x = nn::add(&x, &cpu_linear(&f, &b.f2, ROWS));
    }
    let logits = cpu_linear(&nn::layernorm(&x, &lnf.0, &lnf.1, ROWS, C), &head, ROWS);
    let (want_loss, want_grad) = nn::cross_entropy(&logits, &targets, VOCAB);

    // GPU: every op except the matmul reads and writes device buffers; the matmul inputs are
    // read back so the composition under test is the elementwise/attention/loss chain.
    let rd = |b: &GpuBuf| g.read(b).unwrap();
    let (gt, gp, gi) = (
        g.upload(&tok).unwrap(),
        g.upload(&pos).unwrap(),
        g.upload_indices(&ids, VOCAB).unwrap(),
    );
    let mut gx = g.zeros(ROWS * C).unwrap();
    ops::embed(&g, (&gt, &gp, &gi), &gx, (ROWS, C, T)).unwrap();
    let norm = |g: &Gpu, x: &GpuBuf, p: &(Vec<f32>, Vec<f32>)| {
        let (ga, gb, y) = (
            g.upload(&p.0).unwrap(),
            g.upload(&p.1).unwrap(),
            g.zeros(ROWS * C).unwrap(),
        );
        ops::layernorm(g, x, &ga, &gb, &y, (ROWS, C)).unwrap();
        y
    };
    for b in &blocks {
        let h = norm(&g, &gx, &b.ln1);
        let hh = rd(&h);
        let (q, k, v) = (
            gpu_linear(&g, &h, &b.q, ROWS, &hh),
            gpu_linear(&g, &h, &b.k, ROWS, &hh),
            gpu_linear(&g, &h, &b.v, ROWS, &hh),
        );
        let (probs, a) = (
            g.zeros(BATCH * HEADS * T * T).unwrap(),
            g.zeros(ROWS * C).unwrap(),
        );
        ops::attention(&g, (&q, &k, &v), (&probs, &a), (BATCH, T, HEADS, hd)).unwrap();
        let o = gpu_linear(&g, &a, &b.o, ROWS, &rd(&a));
        let x2 = g.zeros(ROWS * C).unwrap();
        ops::add(&g, &gx, &o, &x2).unwrap();
        let h2 = norm(&g, &x2, &b.ln2);
        let f1 = gpu_linear(&g, &h2, &b.f1, ROWS, &rd(&h2));
        let act = g.zeros(ROWS * HIDDEN).unwrap();
        ops::gelu(&g, &f1, &act).unwrap();
        let f2 = gpu_linear(&g, &act, &b.f2, ROWS, &rd(&act));
        let x3 = g.zeros(ROWS * C).unwrap();
        ops::add(&g, &x2, &f2, &x3).unwrap();
        gx = x3;
    }
    let hf = norm(&g, &gx, &lnf);
    let glogits = gpu_linear(&g, &hf, &head, ROWS, &rd(&hf));
    let (grad, rl, loss) = (
        g.zeros(ROWS * VOCAB).unwrap(),
        g.zeros(ROWS).unwrap(),
        g.zeros(1).unwrap(),
    );
    let gtg = g.upload_indices(&targets, VOCAB).unwrap();
    ops::cross_entropy(&g, &glogits, &gtg, (&grad, &rl, &loss), (ROWS, VOCAB)).unwrap();

    let (got_logits, got_loss, got_grad) = (rd(&glogits), rd(&loss), rd(&grad));
    let e_logits = max_rel_err(&got_logits, &logits);
    let e_loss = max_rel_err(&got_loss, &[want_loss]);
    let e_grad = max_rel_err(&got_grad, &want_grad);
    eprintln!("forward: logits {e_logits:.2e}, loss {e_loss:.2e}, dlogits {e_grad:.2e} (tol {FORWARD:.0e})");
    assert!(e_logits < FORWARD && e_loss < FORWARD && e_grad < FORWARD);
    assert!(want_loss.is_finite() && want_loss > 0.0);
}
