use crate::shapes::{params, sizes};
use ferrum_ref::llama::{Llama, LlamaTrainer};
use ferrum_ref::optim::Adam;
use ferrum_ref::trainer::StepConfig;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama_train::GpuLlamaTrainer;
use std::time::Instant;

const REPS: usize = 5;

pub(crate) fn table(gpu: &Gpu) {
    println!("\n### Fine-tuning step (batch 4 x 32 tokens, f32 AdamW, clip 1.0)\n");
    println!("| model | params | cpu ms/step | gpu ms/step | gpu vs cpu |");
    println!("|---|---|---|---|---|");
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let sc = StepConfig {
        adam: Adam::new(1e-3),
        clip: 1.0,
    };
    for (name, cfg) in sizes() {
        let w = Llama::random(cfg, 2, false, false);
        let (b, t) = (4, 32);
        let ids: Vec<usize> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
        let targets: Vec<usize> = (0..b * t).map(|i| (i * 11 + 5) % cfg.vocab).collect();
        let mut cpu = LlamaTrainer::new(w.clone(), threads);
        let cpu_ms = step_ms(|| {
            cpu.step(&ids, &targets, b, &sc).unwrap();
        });
        let mut tr = GpuLlamaTrainer::new(gpu, &w, None).unwrap();
        let gpu_ms = step_ms(|| {
            tr.step(&ids, &targets, b, &sc).unwrap();
        });
        println!(
            "| {name} | {:.1}M | {cpu_ms:.1} | {gpu_ms:.1} | {:.2}x |",
            params(&cfg, false) as f64 / 1e6,
            cpu_ms / gpu_ms
        );
    }
    println!("\n(CPU baseline: `ferrum_ref::llama` with {threads} matmul threads.)");
}

/// One warm-up step, then the mean of `REPS`.
fn step_ms(mut step: impl FnMut()) -> f64 {
    step();
    let t0 = Instant::now();
    for _ in 0..REPS {
        step();
    }
    t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
}
