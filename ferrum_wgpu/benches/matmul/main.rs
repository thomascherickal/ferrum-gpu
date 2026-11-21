#![forbid(unsafe_code)]
//! Phase 0 spike: GPU matmul variants vs the CPU reference at 1 and N threads.
//! Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench matmul                 # hardware adapter
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench ... --bench matmul
//!   cargo bench ... --bench matmul -- 256 1024                # custom square sizes
//! Prints a markdown table. Per case: CPU 1 thread, CPU N threads, and per GPU variant
//! compute-only (weights resident), +upload of both operands, +readback of the result.

mod cases;
mod variant;

use ferrum_ref::matmul::rand_vec;
use ferrum_ref::tuned::matmul_tuned;
use ferrum_wgpu::bench_util::{gflops, open_gpu, reps_for, time_ms};
use ferrum_wgpu::matmul::{Matmul, Variant};

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![256, 1024, 2048]
    } else {
        sizes
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let Some(gpu) = open_gpu() else { return };
    let info = gpu.info();
    println!(
        "adapter: {} ({}, {:?}); cpu threads: {threads}",
        info.name, info.backend, info.kind
    );
    // Fixed costs: an empty-ish submit and a 4-byte readback.
    let tiny = gpu.upload(&[1.0]).unwrap();
    let sub = time_ms(50, || {
        gpu.roundtrip().unwrap();
    });
    let rb = time_ms(50, || {
        gpu.read(&tiny).unwrap();
    });
    println!("fixed cost: empty submit+wait {sub:.3} ms, 4-byte readback {rb:.3} ms\n");
    println!("| case | variant | cpu x1 ms | cpu x{threads} ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x{threads} (compute+readback) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (name, (m, k, n)) in cases::all(&sizes) {
        let (a, b) = (rand_vec(m * k, 1), rand_vec(k * n, 2));
        let reps = reps_for(m, k, n);
        let cpu1 = time_ms(reps.min(3), || {
            std::hint::black_box(matmul_tuned(&a, &b, m, k, n, 1).unwrap());
        });
        let cpun = time_ms(reps.min(5), || {
            std::hint::black_box(matmul_tuned(&a, &b, m, k, n, threads).unwrap());
        });
        for v in Variant::ALL {
            let mm = Matmul::new(&gpu, v).unwrap();
            let t = variant::measure(&gpu, &mm, (&a, &b), (m, k, n), reps);
            println!(
                "| {name} | {v:?} | {cpu1:.2} | {cpun:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.2}x |",
                t.compute,
                t.with_readback,
                t.with_upload,
                t.with_all,
                gflops(m, k, n, t.compute / 1e3),
                cpun / t.with_readback
            );
        }
    }
}
