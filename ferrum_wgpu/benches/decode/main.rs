#![forbid(unsafe_code)]
//! Phase 3 crossover: cached single-token decode, CPU (one thread, the way ferrum-cpu runs
//! m=1) vs GPU with resident weights and KV cache. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench decode
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench -p ferrum_wgpu --bench decode
//! Prints a markdown table of tokens/s and ms/token.

mod sizes;

use ferrum_ref::gpt2::{Cache, Gpt2};
use ferrum_wgpu::bench_util::open_gpu;
use ferrum_wgpu::gpt2::GpuGpt2;
use std::time::Instant;

fn main() {
    let Some(gpu) = open_gpu() else { return };
    let info = gpu.info();
    println!(
        "adapter: {} ({}, {:?})\n",
        info.name, info.backend, info.kind
    );
    println!(
        "| model | params | cpu ms/token | cpu tok/s | gpu ms/token | gpu tok/s | gpu vs cpu |"
    );
    println!("|---|---|---|---|---|---|---|");
    for (name, cfg) in sizes::all() {
        let w = Gpt2::random(cfg, 1);
        let n = 48.min(cfg.ctx - 8);
        let toks: Vec<usize> = (0..n + 4).map(|i| (i * 7 + 3) % cfg.vocab).collect();

        let mut cache = Cache::new(cfg.layers);
        for &t in &toks[..4] {
            w.feed(&mut cache, t).unwrap();
        }
        let t0 = Instant::now();
        for &t in &toks[4..] {
            std::hint::black_box(w.feed(&mut cache, t).unwrap());
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;

        let mut m = GpuGpt2::new(&gpu, &w, None).unwrap();
        for &t in &toks[..4] {
            m.feed(t).unwrap(); // compiles the pipelines
        }
        m.reset();
        for &t in &toks[..4] {
            m.feed(t).unwrap();
        }
        let t0 = Instant::now();
        for &t in &toks[4..] {
            std::hint::black_box(m.feed(t).unwrap());
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!(
            "| {name} | {:.1}M | {cpu_ms:.3} | {:.0} | {gpu_ms:.3} | {:.0} | {:.2}x |",
            sizes::params(&cfg) as f64 / 1e6,
            1e3 / cpu_ms,
            1e3 / gpu_ms,
            cpu_ms / gpu_ms
        );
    }
}
