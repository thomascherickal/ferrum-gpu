#![forbid(unsafe_code)]
//! Phase 3 crossover: cached single-token decode, CPU (one thread, the way ferrum-cpu runs
//! m=1) vs GPU with resident weights and KV cache. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench decode
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench -p ferrum_wgpu --bench decode
//! Prints a markdown table of tokens/s and ms/token.

use ferrum_ref::gpt2::{Cache, Config, Gpt2};
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::gpt2::GpuGpt2;
use std::time::Instant;

fn params(c: &Config) -> usize {
    let (d, h) = (c.dim, c.hidden);
    c.vocab * d * 2 + c.ctx * d + c.layers * (4 * d * d + 2 * d * h)
}

fn main() {
    let want = std::env::var("FERRUM_GPU_ADAPTER").ok();
    let gpu = match Gpu::open(want.as_deref()) {
        Ok(g) => g,
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to benchmark");
            return;
        }
    };
    println!(
        "adapter: {} ({}, {:?})\n",
        gpu.info().name,
        gpu.info().backend,
        gpu.info().kind
    );
    let sizes = [
        (
            "small (reference preset)",
            Config {
                vocab: 1024,
                ctx: 128,
                dim: 96,
                heads: 4,
                layers: 3,
                hidden: 320,
            },
        ),
        (
            "medium",
            Config {
                vocab: 4096,
                ctx: 256,
                dim: 256,
                heads: 8,
                layers: 6,
                hidden: 1024,
            },
        ),
        (
            "large",
            Config {
                vocab: 8192,
                ctx: 256,
                dim: 512,
                heads: 8,
                layers: 8,
                hidden: 2048,
            },
        ),
    ];
    println!(
        "| model | params | cpu ms/token | cpu tok/s | gpu ms/token | gpu tok/s | gpu vs cpu |"
    );
    println!("|---|---|---|---|---|---|---|");
    for (name, cfg) in sizes {
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
            params(&cfg) as f64 / 1e6,
            1e3 / cpu_ms,
            1e3 / gpu_ms,
            cpu_ms / gpu_ms
        );
    }
}
