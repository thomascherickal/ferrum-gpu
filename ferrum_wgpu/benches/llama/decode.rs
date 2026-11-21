use crate::shapes::{params, sizes};
use ferrum_ref::llama::{Cache, Llama};
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama::GpuLlama;
use ferrum_wgpu::ops_quant::QKind;
use std::time::Instant;

fn per_token_ms(t0: Instant, n: usize) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3 / n as f64
}

pub fn table(gpu: &Gpu) {
    println!("### Decode (single stream, ms/token and tokens/s)\n");
    println!("| model | params | cpu f32 (1 thread) | gpu f32 | gpu Q8_0 | gpu Q4_0 |");
    println!("|---|---|---|---|---|---|");
    for (name, cfg) in sizes() {
        let w = Llama::random(cfg, 1, false, false);
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
        let cpu_ms = per_token_ms(t0, n);
        let mut row = format!(
            "| {name} | {:.1}M | {cpu_ms:.2} ({:.0}/s)",
            params(&cfg, false) as f64 / 1e6,
            1e3 / cpu_ms
        );
        for q in [None, Some(QKind::Q8_0), Some(QKind::Q4_0)] {
            let mut m = GpuLlama::new(gpu, &w, q, None).unwrap();
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
            let ms = per_token_ms(t0, n);
            row += &format!(" | {ms:.2} ({:.0}/s, {:.2}x)", 1e3 / ms, cpu_ms / ms);
        }
        println!("{row} |");
    }
}
