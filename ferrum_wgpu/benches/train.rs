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

use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::grad::loss_and_grads;
use ferrum_ref::optim::{adam_step, clip_scale, global_norm, Adam, AdamState};
use ferrum_wgpu::bench_util::median;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::train::{GpuTrainer, StepConfig};
use std::time::{Duration, Instant};

fn params(c: &Config) -> usize {
    Gpt2::random(Config { layers: 1, ..*c }, 1).num_params()
        + (c.layers - 1) * (4 * c.dim * c.dim + 2 * c.dim * c.hidden + 10 * c.dim + c.hidden)
}

/// Data-parallel CPU step over `shards` shards; returns the mean loss.
#[allow(clippy::too_many_arguments)] // a bench helper; a params struct would only add noise
fn cpu_step(
    w: &mut Gpt2,
    st: &mut AdamState,
    ids: &[usize],
    tg: &[usize],
    batch: usize,
    shards: usize,
    adam: &Adam,
    clip: f32,
) -> f32 {
    let per = batch / shards;
    let t = ids.len() / batch;
    let results: Vec<(f32, Gpt2)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..shards)
            .map(|i| {
                let (w, lo, hi) = (&*w, i * per * t, (i + 1) * per * t);
                s.spawn(move || {
                    loss_and_grads(w, &ids[lo..hi], &tg[lo..hi], per, 1).expect("valid batch")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("shard thread"))
            .collect()
    });
    let n = shards as f32;
    let loss = results.iter().map(|r| r.0).sum::<f32>() / n;
    let mut g = results[0].1.clone();
    for (i, p) in g.params_mut().into_iter().enumerate() {
        for r in &results[1..] {
            for (a, b) in p.iter_mut().zip(r.1.params()[i]) {
                *a += b;
            }
        }
        p.iter_mut().for_each(|a| *a /= n);
    }
    let scale = clip_scale(global_norm(&g), clip);
    adam_step(w, &g, st, adam, scale);
    loss
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f(); // warm-up (also compiles GPU pipelines)
    let mut s: Vec<Duration> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    median(&mut s).as_secs_f64() * 1e3
}

fn main() {
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
        "adapter: {} ({}, {:?}); cpu threads: {threads}\n",
        gpu.info().name,
        gpu.info().backend,
        gpu.info().kind
    );
    let cases = [
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
            16usize,
            128usize,
            10usize,
        ),
        (
            "medium",
            Config {
                vocab: 4096,
                ctx: 128,
                dim: 256,
                heads: 8,
                layers: 4,
                hidden: 1024,
            },
            8,
            128,
            5,
        ),
        (
            "large",
            Config {
                vocab: 4096,
                ctx: 128,
                dim: 512,
                heads: 8,
                layers: 6,
                hidden: 2048,
            },
            8,
            128,
            3,
        ),
    ];
    println!("| model | params | batch x t | cpu x1 ms/step | cpu x{threads} (data-parallel) ms/step | gpu ms/step | gpu+QAT ms/step | gpu tok/s | gpu vs cpu x{threads} |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let adam = Adam {
        weight_decay: 0.01,
        ..Adam::new(0.001)
    };
    for (name, cfg, batch, t, reps) in cases {
        let w0 = Gpt2::init(cfg, 1);
        let ids: Vec<usize> = (0..batch * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
        let tg: Vec<usize> = (0..batch * t).map(|i| (i * 5 + 4) % cfg.vocab).collect();

        let (mut w, mut st) = (w0.clone(), AdamState::new(&w0));
        let cpu1 = time(reps.min(3), || {
            std::hint::black_box(cpu_step(&mut w, &mut st, &ids, &tg, batch, 1, &adam, 1.0));
        });
        let shards = threads.min(batch);
        let (mut w, mut st) = (w0.clone(), AdamState::new(&w0));
        let cpun = time(reps, || {
            std::hint::black_box(cpu_step(
                &mut w, &mut st, &ids, &tg, batch, shards, &adam, 1.0,
            ));
        });

        let mut tr = match GpuTrainer::new(&gpu, &w0, None) {
            Ok(t) => t,
            Err(e) => {
                println!("| {name} | {:.1}M | {batch} x {t} | {cpu1:.1} | {cpun:.1} | refused: {e} | | | |", params(&cfg) as f64 / 1e6);
                continue;
            }
        };
        let sc = StepConfig { adam, clip: 1.0 };
        let gpu_ms = time(reps, || {
            std::hint::black_box(tr.step(&ids, &tg, batch, &sc).expect("gpu step"));
        });
        // The same trainer with quantisation-aware training on (one Q8_0 snap pass per step).
        tr.set_qat(true).expect("qat buffers");
        let qat_ms = time(reps, || {
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
