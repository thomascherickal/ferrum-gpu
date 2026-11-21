#![forbid(unsafe_code)]
//! Per-kernel timings at the "large" training shapes (batch 8 x 128 tokens, dim 512, hidden
//! 2048, vocab 4096, 8 heads), to see where a training step spends its time. Each kernel runs
//! alone, resident, submit + wait. "per step" counts how often a 6-layer step runs it.
//!   FERRUM_GPU_ADAPTER=MX330 cargo bench -p ferrum_wgpu --bench profile_train

mod attention;
mod loss_and_update;
mod matmuls;
mod pointwise;
mod rig;

use ferrum_wgpu::bench_util::open_gpu;
use rig::{Dims, Rig};

fn main() {
    let Some(g) = open_gpu() else { return };
    println!(
        "adapter: {} ({}, {:?})\n",
        g.info().name,
        g.info().backend,
        g.info().kind
    );
    let d = Dims {
        b: 8,
        t: 128,
        c: 512,
        h: 2048,
        v: 4096,
        heads: 8,
        layers: 6,
    };
    let mut rig = Rig::new(g, d);
    matmuls::rows(&mut rig);
    attention::rows(&mut rig);
    pointwise::rows(&mut rig);
    loss_and_update::loss_rows(&mut rig);
    loss_and_update::optimizer_rows(&mut rig);
    rig.print_table();
}
