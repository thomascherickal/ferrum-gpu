#![forbid(unsafe_code)]
//! Phase 4 gate: one full training step (forward, backward, clip, AdamW) on the CPU vs the GPU.
//! Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench train
//!   FERRUM_GPU_ADAPTER=llvmpipe cargo bench -p ferrum_wgpu --bench train
//!
//! CPU baselines (both use the tuned matmul):
//!   * 1 thread: one shard.
//!   * N threads: data parallel like ferrum-cpu's trainer: the batch is split into N shards,
//!     each thread runs forward+backward on its shard with its own activations, the shard
//!     gradients are averaged, then one AdamW step. This is a stronger baseline than splitting
//!     only the matmuls, because every op scales with the thread count.

mod cases;
mod cpu_step;

use cpu_step::{cpu_step, params, Batch};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::{Adam, AdamState};
use ferrum_wgpu::bench_util::{open_gpu, time_ms};
use ferrum_wgpu::train::{GpuTrainer, StepConfig};

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let Some(gpu) = open_gpu() else { return };
    let info = gpu.info();
    println!(
        "adapter: {} ({}, {:?}); cpu threads: {threads}\n",
        info.name, info.backend, info.kind
    );
    println!("| model | params | batch x t | cpu x1 ms/step | cpu x{threads} (data-parallel) ms/step | gpu ms/step | gpu+QAT ms/step | gpu tok/s | gpu vs cpu x{threads} |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let adam = Adam {
        weight_decay: 0.01,
        ..Adam::new(0.001)
    };
    for (name, cfg, batch, t, reps) in cases::all() {
        let w0 = Gpt2::init(cfg, 1);
        let ids: Vec<usize> = (0..batch * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
        let tg: Vec<usize> = (0..batch * t).map(|i| (i * 5 + 4) % cfg.vocab).collect();
        let b = Batch {
            ids: &ids,
            targets: &tg,
            size: batch,
        };
        let (mut w, mut st) = (w0.clone(), AdamState::new(&w0));
        let cpu1 = time_ms(reps.min(3), || {
            std::hint::black_box(cpu_step(&mut w, &mut st, &b, 1, &adam));
        });
        let shards = threads.min(batch);
        let (mut w, mut st) = (w0.clone(), AdamState::new(&w0));
        let cpun = time_ms(reps, || {
            std::hint::black_box(cpu_step(&mut w, &mut st, &b, shards, &adam));
        });
        let mut tr = match GpuTrainer::new(&gpu, &w0, None) {
            Ok(t) => t,
            Err(e) => {
                println!("| {name} | {:.1}M | {batch} x {t} | {cpu1:.1} | {cpun:.1} | refused: {e} | | | |", params(&cfg) as f64 / 1e6);
                continue;
            }
        };
        let sc = StepConfig { adam, clip: 1.0 };
        let gpu_ms = time_ms(reps, || {
            std::hint::black_box(tr.step(&ids, &tg, batch, &sc).expect("gpu step"));
        });
        // The same trainer with quantisation-aware training on (one Q8_0 snap pass per step).
        tr.set_qat(true).expect("qat buffers");
        let qat_ms = time_ms(reps, || {
            std::hint::black_box(tr.step(&ids, &tg, batch, &sc).expect("gpu qat step"));
        });
        println!(
            "| {name} | {:.1}M | {batch} x {t} | {cpu1:.1} | {cpun:.1} | {gpu_ms:.1} | {qat_ms:.1} | {:.0} | {:.2}x |",
            w0.num_params() as f64 / 1e6,
            (batch * t) as f64 / (gpu_ms / 1e3),
            cpun / gpu_ms
        );
    }
}
