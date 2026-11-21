#![forbid(unsafe_code)]
//! How big a model does this adapter actually run? Real-size shapes (random weights: speed and
//! memory depend on the shape, not the values; nothing is downloaded), each at f32, Q8_0 and
//! Q4_0. A row is either measured (single-stream decode, GPU only) or `refused` with the
//! reason the memory plan gives. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench capability
//!   FERRUM_GPU_ADAPTER="Iris(R) Xe" cargo bench -p ferrum_wgpu --bench capability

mod decode;
mod row;
mod shape_rows;
mod shapes;

use decode::NEW_TOKENS;
use ferrum_wgpu::bench_util::open_gpu;
use ferrum_wgpu::ops_quant::QKind;
use shape_rows::shape_rows;

const PRECISIONS: [(&str, Option<QKind>); 3] = [
    ("f32", None),
    ("Q8_0", Some(QKind::Q8_0)),
    ("Q4_0", Some(QKind::Q4_0)),
];

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
    println!(
        "Single-stream decode, {NEW_TOKENS} tokens after a 4-token prompt, context 2048 planned.\n"
    );
    println!("| model | params | precision | plan MiB | ms/token | tokens/s | note |");
    println!("|---|---|---|---|---|---|---|");
    // a failed load can leave device buffers allocated until the process ends, so the driver
    // script runs one (shape, precision) per process with these two filters
    let only = std::env::var("FERRUM_CAP_ONLY").ok();
    let only_prec = std::env::var("FERRUM_CAP_PREC").ok();
    for s in shapes::all()
        .iter()
        .filter(|s| only.as_deref().is_none_or(|o| s.name.contains(o)))
    {
        shape_rows(&gpu, s, only_prec.as_deref());
    }
}
