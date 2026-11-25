use crate::shapes::{params, sizes};
use ferrum_ref::llama::{Cache, Llama};
use ferrum_wgpu::bench_util::{decode_ms, decode_tokens, DECODE_PRIME};
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama::GpuLlama;
use ferrum_wgpu::ops_quant::QKind;

pub(crate) fn table(gpu: &Gpu) {
    println!("### Decode (single stream, ms/token and tokens/s)\n");
    println!("| model | params | cpu f32 (1 thread) | gpu f32 | gpu Q8_0 | gpu Q4_0 |");
    println!("|---|---|---|---|---|---|");
    for (name, cfg) in sizes() {
        let w = Llama::random(cfg, 1, false, false);
        let toks = decode_tokens(cfg.ctx, cfg.vocab);
        let mut cache = Cache::new(cfg.layers);
        let cpu_ms = decode_ms(&toks, |t| {
            std::hint::black_box(w.feed(&mut cache, t).unwrap());
        });
        let mut row = format!(
            "| {name} | {:.1}M | {cpu_ms:.2} ({:.0}/s)",
            params(&cfg, false) as f64 / 1e6,
            1e3 / cpu_ms
        );
        for q in [None, Some(QKind::Q8_0), Some(QKind::Q4_0)] {
            let mut m = GpuLlama::new(gpu, &w, q, None).unwrap();
            for &t in &toks[..DECODE_PRIME] {
                m.feed(t).unwrap(); // compiles the pipelines
            }
            m.reset();
            let ms = decode_ms(&toks, |t| {
                std::hint::black_box(m.feed(t).unwrap());
            });
            row += &format!(" | {ms:.2} ({:.0}/s, {:.2}x)", 1e3 / ms, cpu_ms / ms);
        }
        println!("{row} |");
    }
}
