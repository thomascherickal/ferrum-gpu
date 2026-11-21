#![forbid(unsafe_code)]
//! Phase 5 gate: Llama-family decode and fine-tuning, CPU vs GPU, plus the size table from the
//! buffer plans. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench llama
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench -p ferrum_wgpu --bench llama
//! Weights are random: speed and memory depend on the shape, not the values. Decode is
//! single-stream (the CPU runs one thread, the way a CPU runtime runs m=1); training uses all
//! logical CPUs for the CPU baseline.

mod decode;
mod finetune;
mod plans;
mod shapes;

use ferrum_wgpu::bench_util::open_gpu;

fn main() {
    let Some(gpu) = open_gpu() else { return };
    let info = gpu.info();
    println!(
        "adapter: {} ({}, {:?}), max binding {} MiB\n",
        info.name,
        info.backend,
        info.kind,
        info.max_binding / (1024 * 1024)
    );
    decode::table(&gpu);
    finetune::table(&gpu);
    plans::table(&gpu);
}
