#![forbid(unsafe_code)]
//! Per-kernel timings at the "large" training shapes (batch 8 x 128 tokens, dim 512, hidden
//! 2048, vocab 4096, 8 heads), to see where a training step spends its time. Each kernel runs
//! alone, resident, submit + wait. "per step" counts how often a 6-layer step runs it.
//!   FERRUM_GPU_ADAPTER=MX330 cargo bench -p ferrum_wgpu --bench profile_train

use ferrum_ref::matmul::rand_vec;
use ferrum_wgpu::bench_util::median;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::ops_bwd as bwd;
use ferrum_wgpu::ops_opt;
use ferrum_wgpu::{matmul, ops};
use std::time::{Duration, Instant};

fn ms<F: FnMut()>(mut f: F) -> f64 {
    f();
    let mut s: Vec<Duration> = (0..7)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    median(&mut s).as_secs_f64() * 1e3
}

fn main() {
    let want = std::env::var("FERRUM_GPU_ADAPTER").ok();
    let g = match Gpu::open(want.as_deref()) {
        Ok(g) => g,
        Err(e) => {
            println!("no GPU adapter ({e})");
            return;
        }
    };
    println!(
        "adapter: {} ({}, {:?})\n",
        g.info().name,
        g.info().backend,
        g.info().kind
    );
    let (b, t, c, h, v, heads, layers) = (
        8usize, 128usize, 512usize, 2048usize, 4096usize, 8usize, 6usize,
    );
    let (r, hd) = (b * t, c / heads);
    let buf = |n: usize| g.upload(&rand_vec(n, n as u64)).unwrap();
    let mut rows: Vec<(String, f64, usize)> = Vec::new();
    let mut add =
        |name: &str, per_step: usize, t_ms: f64| rows.push((name.to_string(), t_ms, per_step));

    // Matmuls: forward NN, and the two backward flavours (dx = dy w^T, dw = x^T dy).
    for (name, m, k, n, per) in [
        ("matmul NN  qkvo   [1024x512x512]", r, c, c, 4 * layers),
        ("matmul NN  ffn up [1024x512x2048]", r, c, h, layers),
        ("matmul NN  ffn dn [1024x2048x512]", r, h, c, layers),
        ("matmul NN  head   [1024x512x4096]", r, c, v, 1),
    ] {
        let (a, w, o) = (buf(m * k), buf(k * n), g.zeros(m * n).unwrap());
        let mm = matmul::Matmul::new(&g, matmul::Variant::Blocked4).unwrap();
        let t_ms = ms(|| {
            mm.submit(&g, &a, &w, &o, (m, k, n)).unwrap();
            g.wait().unwrap();
        });
        add(name, per, t_ms);
        // dx: dy[m,n] * w^T -> [m,k]; dw: x^T[k,m] * dy[m,n] -> [k,n]
        let (dy, dx, dw) = (buf(m * n), g.zeros(m * k).unwrap(), g.zeros(k * n).unwrap());
        let t_dx = ms(|| {
            bwd::matmul_ex(&g, (&dy, &w, &dx), (m, n, k), (false, true)).unwrap();
            g.wait().unwrap();
        });
        let t_dw = ms(|| {
            bwd::matmul_ex(&g, (&a, &dy, &dw), (k, m, n), (true, false)).unwrap();
            g.wait().unwrap();
        });
        add(
            &name
                .replace("NN ", "dX ")
                .replace("matmul dX ", "matmul dX"),
            per,
            t_dx,
        );
        add(
            &name
                .replace("NN ", "dW ")
                .replace("matmul dW ", "matmul dW"),
            per,
            t_dw,
        );
    }

    // Attention forward and backward.
    let n = r * c;
    let (q, k, vv) = (buf(n), buf(n), buf(n));
    let (probs, ao) = (g.zeros(b * heads * t * t).unwrap(), g.zeros(n).unwrap());
    add(
        "attention forward",
        layers,
        ms(|| {
            ops::attention(&g, (&q, &k, &vv), (&probs, &ao), (b, t, heads, hd)).unwrap();
            g.wait().unwrap();
        }),
    );
    let (dout, dq, dk, dv, dot, dsc) = (
        buf(n),
        g.zeros(n).unwrap(),
        g.zeros(n).unwrap(),
        g.zeros(n).unwrap(),
        g.zeros(b * heads * t).unwrap(),
        g.zeros(b * heads * t * t).unwrap(),
    );
    add(
        "attention backward",
        layers,
        ms(|| {
            bwd::attention_bwd(
                &g,
                (&q, &k, &vv, &probs, &dout),
                (&dq, &dk, &dv),
                (&dot, &dsc),
                (b, t, heads, hd),
            )
            .unwrap();
            g.wait().unwrap();
        }),
    );

    // LayerNorm forward and backward, GELU, bias, residual add.
    let (x, gain, shift, y, st) = (
        buf(r * c),
        buf(c),
        buf(c),
        g.zeros(r * c).unwrap(),
        g.zeros(2 * r).unwrap(),
    );
    add(
        "layernorm forward",
        2 * layers + 1,
        ms(|| {
            ops::layernorm(&g, &x, &gain, &shift, &y, (r, c)).unwrap();
            g.wait().unwrap();
        }),
    );
    let (dx, dg, db) = (
        g.zeros(r * c).unwrap(),
        g.zeros(c).unwrap(),
        g.zeros(c).unwrap(),
    );
    add(
        "layernorm backward",
        2 * layers + 1,
        ms(|| {
            bwd::layernorm_bwd(&g, (&x, &gain, &y), (&dx, &dg, &db), &st, (r, c)).unwrap();
            g.wait().unwrap();
        }),
    );
    let (f1, act) = (buf(r * h), g.zeros(r * h).unwrap());
    add(
        "gelu forward",
        layers,
        ms(|| {
            ops::gelu(&g, &f1, &act).unwrap();
            g.wait().unwrap();
        }),
    );
    let (da, df) = (buf(r * h), g.zeros(r * h).unwrap());
    add(
        "gelu backward",
        layers,
        ms(|| {
            bwd::gelu_bwd(&g, &da, &f1, &df).unwrap();
            g.wait().unwrap();
        }),
    );
    let (bias, yb) = (buf(c), g.zeros(r * c).unwrap());
    add(
        "bias add [1024x512]",
        5 * layers,
        ms(|| {
            ops::add_bias(&g, &x, &bias, &yb).unwrap();
            g.wait().unwrap();
        }),
    );
    let (xa, xb, xc) = (buf(r * c), buf(r * c), g.zeros(r * c).unwrap());
    add(
        "residual add [1024x512]",
        2 * layers + 2 * layers + 2,
        ms(|| {
            ops::add(&g, &xa, &xb, &xc).unwrap();
            g.wait().unwrap();
        }),
    );
    // Column sums (bias gradients), the largest being the FFN-up bias and the junk head bias.
    for (name, cols, per) in [
        ("colsum [1024x512] (bias grad)", c, 5 * layers),
        ("colsum [1024x2048] (ffn bias grad)", h, layers),
        ("colsum [1024x4096] (head, unused result)", v, 1),
    ] {
        let (xin, out) = (buf(r * cols), g.zeros(cols).unwrap());
        add(
            name,
            per,
            ms(|| {
                bwd::colsum(&g, &xin, &out, (r, cols)).unwrap();
                g.wait().unwrap();
            }),
        );
    }
    // Loss.
    let (logits, grad, rl, loss) = (
        buf(r * v),
        g.zeros(r * v).unwrap(),
        g.zeros(r).unwrap(),
        g.zeros(1).unwrap(),
    );
    let tg = g
        .upload_indices(&(0..r).map(|i| i % v).collect::<Vec<_>>(), v)
        .unwrap();
    add(
        "cross-entropy [1024x4096]",
        1,
        ms(|| {
            ops::cross_entropy(&g, &logits, &tg, (&grad, &rl, &loss), (r, v)).unwrap();
            g.wait().unwrap();
        }),
    );
    // Embedding gradient.
    let ids: Vec<usize> = (0..r).map(|i| (i * 7 + 3) % v).collect();
    let (dtok, dpos) = (g.zeros(v * c).unwrap(), g.zeros(t * c).unwrap());
    add(
        "embedding gradient",
        1,
        ms(|| {
            bwd::embed_bwd(&g, &x, &ids, (&dtok, &dpos), (v, c, t, t)).unwrap();
            g.wait().unwrap();
        }),
    );
    // Optimizer over a 3.15M-element block (one transformer layer's worth, as a stand-in) .
    let (p, gr, mo, ve) = (
        buf(h * c),
        buf(h * c),
        g.zeros(h * c).unwrap(),
        g.zeros(h * c).unwrap(),
    );
    let scale = g.upload(&[1.0, 0.0]).unwrap();
    let adam = ferrum_ref::optim::Adam::new(0.001);
    add(
        "adamw update [2048x512 tensor]",
        1 + layers * 2,
        ms(|| {
            ops_opt::adam_update(&g, (&p, &gr, &mo, &ve), &scale, &adam, true, 1).unwrap();
            g.wait().unwrap();
        }),
    );
    let part = g.zeros(ops_opt::parts_for(h * c)).unwrap();
    add(
        "grad sum-of-squares [2048x512 tensor]",
        1 + layers * 2,
        ms(|| {
            ops_opt::sumsq_parts(&g, &gr, &part, 0).unwrap();
            g.wait().unwrap();
        }),
    );

    println!("| kernel | ms each | per step | ms per step | share |");
    println!("|---|---|---|---|---|");
    let total: f64 = rows.iter().map(|r| r.1 * r.2 as f64).sum();
    for (name, t_ms, per) in &rows {
        println!(
            "| {name} | {t_ms:.3} | {per} | {:.1} | {:.0}% |",
            t_ms * *per as f64,
            100.0 * t_ms * *per as f64 / total
        );
    }
    println!("| **estimated sum** | | | **{total:.0}** | |");
}
