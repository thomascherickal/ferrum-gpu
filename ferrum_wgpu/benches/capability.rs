#![forbid(unsafe_code)]
//! How big a model does this adapter actually run? Real-size shapes (random weights: speed and
//! memory depend on the shape, not the values; nothing is downloaded), each at f32, Q8_0 and
//! Q4_0. A row is either measured (single-stream decode, GPU only) or `refused` with the
//! reason the memory plan gives. Std-only (`harness = false`).
//!   cargo bench -p ferrum_wgpu --bench capability
//!   FERRUM_GPU_ADAPTER="Iris(R) Xe" cargo bench -p ferrum_wgpu --bench capability

use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama::{plan, GpuLlama};
use ferrum_wgpu::llama_train::param_sizes;
use ferrum_wgpu::ops_quant::QKind;
use std::time::Instant;

const NEW_TOKENS: usize = 16;

struct Shape {
    name: &'static str,
    cfg: Config,
    biases: bool,
    tied: bool,
}

#[allow(clippy::too_many_arguments)]
fn shape(
    name: &'static str,
    (vocab, ctx, dim): (usize, usize, usize),
    (heads, kv, hd): (usize, usize, usize),
    (layers, hidden): (usize, usize),
    (biases, tied): (bool, bool),
) -> Shape {
    Shape {
        name,
        cfg: Config {
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
        },
        biases,
        tied,
    }
}

/// `MemAvailable` in bytes, from `/proc/meminfo` (Linux); `None` elsewhere.
fn mem_available() -> Option<u64> {
    // the kernel's memory report: system information, not a file of this project or of ferrum-cpu
    let meminfo = std::path::Path::new("/proc").join("meminfo");
    let text = std::fs::read_to_string(meminfo).ok()?;
    let kb: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb * 1024)
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
    println!(
        "Single-stream decode, {NEW_TOKENS} tokens after a 4-token prompt, context 2048 planned.\n"
    );
    println!("| model | params | precision | plan MiB | ms/token | tokens/s | note |");
    println!("|---|---|---|---|---|---|---|");
    let shapes = [
        shape(
            "SmolLM-135M shape",
            (49152, 2048, 576),
            (9, 3, 64),
            (30, 1536),
            (false, true),
        ),
        shape(
            "SmolLM-360M shape",
            (49152, 2048, 960),
            (15, 5, 64),
            (32, 2560),
            (false, true),
        ),
        shape(
            "Qwen2-0.5B shape",
            (151936, 2048, 896),
            (14, 2, 64),
            (24, 4864),
            (true, true),
        ),
        shape(
            "TinyLlama-1.1B shape",
            (32000, 2048, 2048),
            (32, 4, 64),
            (22, 5632),
            (false, false),
        ),
        shape(
            "Qwen2.5-1.5B shape",
            (151936, 2048, 1536),
            (12, 2, 128),
            (28, 8960),
            (true, true),
        ),
    ];
    // a failed load can leave device buffers allocated until the process ends, so the driver
    // script runs one (shape, precision) per process with these two filters
    let only = std::env::var("FERRUM_CAP_ONLY").ok();
    let only_prec = std::env::var("FERRUM_CAP_PREC").ok();
    for s in &shapes {
        if only.as_deref().is_some_and(|o| !s.name.contains(o)) {
            continue;
        }
        let n_params: usize = param_sizes(&s.cfg, s.biases).iter().sum();
        let params = format!("{:.0}M", n_params as f64 / 1e6);
        // build the weights once per shape, and only if some precision fits
        let precisions = [
            ("f32", None),
            ("Q8_0", Some(QKind::Q8_0)),
            ("Q4_0", Some(QKind::Q4_0)),
        ];
        let fits: Vec<bool> = precisions
            .iter()
            .map(|(_, q)| plan(&s.cfg, *q).check(info.max_binding, None).is_ok())
            .collect();
        // random f32 weights are built on the host first: keep 1.5x their size free so a big
        // shape cannot push the machine into swap or the OOM killer
        let need = n_params as u64 * 4 * 3 / 2;
        if fits.iter().any(|&f| f) && mem_available().is_some_and(|a| a < need) {
            println!(
                "| {} | {params} | all | - | - | - | skipped: needs {} MiB of free host RAM for the f32 weights, {} MiB available |",
                s.name,
                mib(need),
                mib(mem_available().unwrap_or(0))
            );
            continue;
        }
        let w = fits
            .iter()
            .any(|&f| f)
            .then(|| Llama::random(s.cfg, 1, s.biases, s.tied));
        for ((label, q), fit) in precisions.iter().zip(&fits) {
            if only_prec.as_deref().is_some_and(|o| o != *label) {
                continue;
            }
            let p = plan(&s.cfg, *q);
            let total = mib(p.total());
            // an integrated GPU's memory is the machine's RAM: its buffers count against it too
            let shared = format!("{:?}", info.kind) == "Integrated"
                && mem_available().is_some_and(|a| a < need + p.total());
            if shared && *fit {
                println!(
                    "| {} | {params} | {label} | {total} | - | - | skipped: integrated GPU shares RAM; weights on the host plus the plan exceed the {} MiB available |",
                    s.name,
                    mib(mem_available().unwrap_or(0))
                );
                continue;
            }
            if !fit {
                let why = p
                    .check(info.max_binding, None)
                    .err()
                    .map_or(String::new(), |e| e.to_string());
                println!(
                    "| {} | {params} | {label} | {total} | - | - | refused: {why} |",
                    s.name
                );
                continue;
            }
            let w = w.as_ref().expect("weights exist when something fits");
            let mut m = match GpuLlama::new(&gpu, w, *q, None) {
                Ok(m) => m,
                Err(e) => {
                    println!(
                        "| {} | {params} | {label} | {total} | - | - | failed to load: {e} |",
                        s.name
                    );
                    continue;
                }
            };
            let toks: Vec<usize> = (0..4 + NEW_TOKENS)
                .map(|i| (i * 7 + 3) % s.cfg.vocab)
                .collect();
            let run = |m: &mut GpuLlama| -> Result<f64, String> {
                for &t in &toks[..4] {
                    m.feed(t).map_err(|e| e.to_string())?;
                }
                let t0 = Instant::now();
                for &t in &toks[4..] {
                    std::hint::black_box(m.feed(t).map_err(|e| e.to_string())?);
                }
                Ok(t0.elapsed().as_secs_f64() * 1e3 / NEW_TOKENS as f64)
            };
            // the first pass compiles the pipelines; the second is the one reported
            let warm = run(&mut m);
            m.reset();
            match warm.and_then(|_| run(&mut m)) {
                Ok(ms) => println!(
                    "| {} | {params} | {label} | {total} | {ms:.1} | {:.1} | |",
                    s.name,
                    1e3 / ms
                ),
                Err(e) => println!(
                    "| {} | {params} | {label} | {total} | - | - | failed: {e} |",
                    s.name
                ),
            }
        }
    }
}
