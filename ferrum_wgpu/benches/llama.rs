#![forbid(unsafe_code)]
//! Phase 5 gate: Llama-family decode and fine-tuning, CPU vs GPU, plus the size table from the
//! buffer plans. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench llama
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench -p ferrum_wgpu --bench llama
//! Weights are random: speed and memory depend on the shape, not the values. Decode is
//! single-stream (the CPU runs one thread, the way a CPU runtime runs m=1); training uses all
//! logical CPUs for the CPU baseline.

use ferrum_ref::llama::{Cache, Config, Llama, LlamaTrainer, RopeType};
use ferrum_ref::optim::Adam;
use ferrum_ref::trainer::StepConfig;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama::{plan, GpuLlama};
use ferrum_wgpu::llama_train::{activation_plan, param_sizes, static_plan_for, GpuLlamaTrainer};
use ferrum_wgpu::ops_quant::QKind;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn shape(
    vocab: usize,
    ctx: usize,
    dim: usize,
    heads: usize,
    kv: usize,
    hd: usize,
    layers: usize,
    hidden: usize,
) -> Config {
    Config {
        vocab,
        ctx,
        dim,
        heads,
        kv_heads: kv,
        head_dim: hd,
        layers,
        hidden,
        rope_dim: hd,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

fn params(c: &Config, biases: bool) -> usize {
    param_sizes(c, biases).iter().sum()
}

fn mib(bytes: u64) -> String {
    format!("{:.0}", bytes as f64 / (1024.0 * 1024.0))
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
    let info = gpu.info();
    println!(
        "adapter: {} ({}, {:?}), max binding {} MiB\n",
        info.name,
        info.backend,
        info.kind,
        info.max_binding / (1024 * 1024)
    );
    let sizes = [
        ("small", shape(1024, 128, 96, 4, 2, 24, 3, 256)),
        ("medium", shape(4096, 256, 256, 8, 2, 32, 6, 704)),
        ("large", shape(8192, 256, 512, 8, 4, 64, 8, 1408)),
    ];

    println!("### Decode (single stream, ms/token and tokens/s)\n");
    println!("| model | params | cpu f32 (1 thread) | gpu f32 | gpu Q8_0 | gpu Q4_0 |");
    println!("|---|---|---|---|---|---|");
    for (name, cfg) in sizes {
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
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let mut row = format!(
            "| {name} | {:.1}M | {cpu_ms:.2} ({:.0}/s)",
            params(&cfg, false) as f64 / 1e6,
            1e3 / cpu_ms
        );
        for q in [None, Some(QKind::Q8_0), Some(QKind::Q4_0)] {
            let mut m = GpuLlama::new(&gpu, &w, q, None).unwrap();
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
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            row += &format!(" | {ms:.2} ({:.0}/s, {:.2}x)", 1e3 / ms, cpu_ms / ms);
        }
        println!("{row} |");
    }

    println!("\n### Fine-tuning step (batch 4 x 32 tokens, f32 AdamW, clip 1.0)\n");
    println!("| model | params | cpu ms/step | gpu ms/step | gpu vs cpu |");
    println!("|---|---|---|---|---|");
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let sc = StepConfig {
        adam: Adam::new(1e-3),
        clip: 1.0,
    };
    for (name, cfg) in sizes {
        let w = Llama::random(cfg, 2, false, false);
        let (b, t) = (4, 32);
        let ids: Vec<usize> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
        let targets: Vec<usize> = (0..b * t).map(|i| (i * 11 + 5) % cfg.vocab).collect();
        let mut cpu = LlamaTrainer::new(w.clone(), threads);
        cpu.step(&ids, &targets, b, &sc).unwrap();
        let reps = 5;
        let t0 = Instant::now();
        for _ in 0..reps {
            cpu.step(&ids, &targets, b, &sc).unwrap();
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let mut tr = GpuLlamaTrainer::new(&gpu, &w, None).unwrap();
        tr.step(&ids, &targets, b, &sc).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            tr.step(&ids, &targets, b, &sc).unwrap();
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!(
            "| {name} | {:.1}M | {cpu_ms:.1} | {gpu_ms:.1} | {:.2}x |",
            params(&cfg, false) as f64 / 1e6,
            cpu_ms / gpu_ms
        );
    }
    println!("\n(CPU baseline: `ferrum_ref::llama` with {threads} matmul threads.)");

    println!("\n### Memory plans (MiB) against this adapter's binding limit\n");
    println!("| model | params | infer f32 | infer Q8_0 | infer Q4_0 | fine-tune f32 (batch 1 x 128) | verdict |");
    println!("|---|---|---|---|---|---|---|");
    let refs = [
        ("small (run above)", sizes[0].1, false),
        ("medium (run above)", sizes[1].1, false),
        ("large (run above)", sizes[2].1, false),
        (
            "SmolLM-135M shape (plan only)",
            shape(49152, 2048, 576, 9, 3, 64, 30, 1536),
            false,
        ),
        (
            "Qwen2-0.5B shape (plan only)",
            shape(151936, 2048, 896, 14, 2, 64, 24, 4864),
            true,
        ),
        (
            "TinyLlama-1.1B shape (plan only)",
            shape(32000, 2048, 2048, 32, 4, 64, 22, 5632),
            false,
        ),
    ];
    for (name, c, biases) in refs {
        let inf = |q| plan(&c, q);
        let t = 128.min(c.ctx);
        let mut ft = static_plan_for(&param_sizes(&c, biases));
        activation_plan(&c, 1, t)
            .rows()
            .iter()
            .for_each(|(n, b)| ft.add(n, *b));
        let verdict = |p: &ferrum_wgpu::budget::BufferPlan| match p.check(info.max_binding, None) {
            Ok(()) => "fits".to_string(),
            Err(e) => format!("refused: {e}"),
        };
        println!(
            "| {name} | {:.0}M | {} | {} | {} | {} | infer f32 {}; infer Q4_0 {}; fine-tune {} |",
            params(&c, biases) as f64 / 1e6,
            mib(inf(None).total()),
            mib(inf(Some(QKind::Q8_0)).total()),
            mib(inf(Some(QKind::Q4_0)).total()),
            mib(ft.total()),
            verdict(&inf(None)),
            verdict(&inf(Some(QKind::Q4_0))),
            verdict(&ft),
        );
    }
}
