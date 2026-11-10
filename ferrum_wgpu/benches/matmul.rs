#![forbid(unsafe_code)]
//! Phase 0 spike: GPU matmul variants vs the CPU reference at 1 and N threads.
//! Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench matmul                 # hardware adapter
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench ... --bench matmul
//!   cargo bench ... --bench matmul -- 256 1024                # custom square sizes
//! Prints a markdown table. Per case: CPU 1 thread, CPU N threads, and per GPU variant
//! compute-only (weights resident), +upload of both operands, +readback of the result.

use ferrum_ref::matmul::{matmul_threaded, rand_vec};
use ferrum_wgpu::bench_util::{gflops, reps_for, time_ms};
use ferrum_wgpu::device::Gpu;
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
    let want = std::env::var("FERRUM_GPU_ADAPTER").ok();
    let gpu = match Gpu::open(want.as_deref()) {
        Ok(g) => g,
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to benchmark");
            return;
        }
    };
    println!(
        "adapter: {} ({}, {:?}); cpu threads: {threads}",
        gpu.info().name,
        gpu.info().backend,
        gpu.info().kind
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

    let mut cases: Vec<(String, (usize, usize, usize))> = sizes
        .iter()
        .map(|&s| (format!("{s}^3"), (s, s, s)))
        .collect();
    // Reference model (embed 96, hidden 320, vocab 1024) at 16 sequences x 128 tokens.
    for (name, k, n) in [
        ("qkv", 96, 96),
        ("ffn1", 96, 320),
        ("ffn2", 320, 96),
        ("head", 96, 1024),
    ] {
        cases.push((format!("ref-{name} (2048x{k}x{n})"), (2048, k, n)));
    }

    println!("| case | variant | cpu x1 ms | cpu x{threads} ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x{threads} (compute+readback) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (name, (m, k, n)) in cases {
        let (a, b) = (rand_vec(m * k, 1), rand_vec(k * n, 2));
        let reps = reps_for(m, k, n);
        let cpu1 = time_ms(reps.min(3), || {
            std::hint::black_box(matmul_threaded(&a, &b, m, k, n, 1).unwrap());
        });
        let cpun = time_ms(reps.min(5), || {
            std::hint::black_box(matmul_threaded(&a, &b, m, k, n, threads).unwrap());
        });
        for v in Variant::ALL {
            let mm = Matmul::new(&gpu, v);
            let (ab, bb, cb) = (
                gpu.upload(&a).unwrap(),
                gpu.upload(&b).unwrap(),
                gpu.zeros(m * n).unwrap(),
            );
            let compute = time_ms(reps, || {
                mm.submit(&gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
                gpu.wait().unwrap();
            });
            let res_read = time_ms(reps, || {
                mm.submit(&gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
                std::hint::black_box(gpu.read(&cb).unwrap());
            });
            let with_up = time_ms(reps, || {
                let (ab, bb) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
                mm.submit(&gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
                gpu.wait().unwrap();
            });
            let with_all = time_ms(reps, || {
                let (ab, bb) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
                mm.submit(&gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
                std::hint::black_box(gpu.read(&cb).unwrap());
            });
            println!(
                "| {name} | {v:?} | {cpu1:.2} | {cpun:.2} | {compute:.2} | {res_read:.2} | {with_up:.2} | {with_all:.2} | {:.1} | {:.2}x |",
                gflops(m, k, n, compute / 1e3),
                cpun / res_read
            );
        }
    }
}
