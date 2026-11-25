#![forbid(unsafe_code)]
//! Phase 3 crossover: cached single-token decode, CPU (one thread, the way ferrum-cpu runs
//! m=1) vs GPU with resident weights and KV cache. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench decode
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench -p ferrum_wgpu --bench decode
//! Prints a markdown table of tokens/s and ms/token.

mod sizes;

use ferrum_ref::gpt2::{Cache, Gpt2};
use ferrum_wgpu::bench_util::{decode_ms, decode_tokens, open_gpu, DECODE_PRIME};
use ferrum_wgpu::gpt2::GpuGpt2;

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
        let toks = decode_tokens(cfg.ctx, cfg.vocab);
        let mut cache = Cache::new(cfg.layers);
        let cpu_ms = decode_ms(&toks, |t| {
            std::hint::black_box(w.feed(&mut cache, t).unwrap());
        });

        let mut m = GpuGpt2::new(&gpu, &w, None).unwrap();
        for &t in &toks[..DECODE_PRIME] {
            m.feed(t).unwrap(); // compiles the pipelines
        }
        m.reset();
        let gpu_ms = decode_ms(&toks, |t| {
            std::hint::black_box(m.feed(t).unwrap());
        });
        println!(
            "| {name} | {:.1}M | {cpu_ms:.3} | {:.0} | {gpu_ms:.3} | {:.0} | {:.2}x |",
            sizes::params(&cfg) as f64 / 1e6,
            1e3 / cpu_ms,
            1e3 / gpu_ms,
            cpu_ms / gpu_ms
        );
    }
}
