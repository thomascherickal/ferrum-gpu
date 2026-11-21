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

mod full_forward_pass_matches_the_cpu_reference;
