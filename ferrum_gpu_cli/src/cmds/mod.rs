//! The commands. Each writes its report to `out` and returns an error string instead of exiting,
//! so tests drive them with the CPU backend and a `Vec<u8>`.

use crate::device::{decide, Workload};
use crate::flags::{val, Args, Flag};

pub mod gpt2;
pub mod runtimes;
use crate::preflight;
use ferrum_gguf::Gguf;
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::{fresh_model, preset, GenOpts, Progress, TrainSpec};
use ferrum_gpu_gui::tokenizer::{decode, encode};
use ferrum_model::checkpoint::atomic_write;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::Llama;
use ferrum_wgpu::Gpu;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// Usage text.
pub const USAGE: &str = "usage: ferrum-gpu <command> [flags]\n\
commands:\n  \
info          [--model F.gguf]\n  \
train         --data FILE --out F.gguf [--arch gpt2|llama] [--preset tiny|small|medium] [--steps N | --epochs N | --tokens N] [--lr X] [--batch N] [--context N] [--seed N] [--quant f32|int8|int4]\n                gpt2 only: --embed --heads --blocks --hidden --vocab --stride --weight_decay --cosine --warmup --clip --no-qat --tokenizer F --checkpoint F --checkpoint-every N --resume F --val X --patience N --max-seconds X --metrics-csv F --verbose --sample\n  \
finetune-gguf --model F.gguf --data FILE --out G.gguf [same training flags]\n  \
generate      --model F.gguf --prompt TEXT [--max-new N] [--temperature X] [--top-k N] [--seed N]\n                gpt2 files also: --top-p X --repetition-penalty X --stop S (repeatable) --stream --ids \"1 2 3\" (aliases --temp --gen-seed --max-tokens --chars)\n  \
run-gguf      same flags as generate; also prints the model's shape first\n  \
prepare       --input FILE|DIR (repeatable) --out DIR [--nfkc] [--lowercase] [--strip-gutenberg] [--dedup] [--min-chars N] [--max-chars N] [--val-frac X] [--test-frac X] [--seed N] [--doc-sep S] [--lossy]   (writes train.txt val.txt test.txt manifest.json)\n  \
tokenize      --data FILE|DIR --tokenizer TOK.gguf --out TOKENS.gguf [--doc-sep S]   (a token file trains with `train --data TOKENS.gguf --tokenizer TOK.gguf`, windows read from disk)\n  \
export-ollama --model F.gguf --name NAME [--bin P] [--modelfile-only OUT]   (GPT-2 files; needs the `ollama` tool)\n  \
export-lmstudio --model F.gguf [--repo publisher/name] [--bin P]   (GPT-2 files; needs the `lms` tool)\n  \
export-gguf   --model F.gguf --out G.gguf [--quant f32|int8|int4]   (re-write a model in another precision; reads Q4_1/Q5_*/Q8_1/K-quant files too)\n  \
eval          --model F.gguf --data FILE [--ctx N] [--json F]\n\
device flags (info, train, finetune-gguf, generate, run-gguf): --device cpu|gpu|auto  --gpu-adapter NAME  --gpu-mem-mb N  --threads N";

const DEVICE: [Flag; 4] = [
    val("--device"),
    val("--gpu-adapter"),
    val("--threads"),
    val("--gpu-mem-mb"),
];
const TRAIN: [Flag; 7] = [
    val("--data"),
    val("--out"),
    val("--steps"),
    val("--lr"),
    val("--batch"),
    val("--context"),
    val("--seed"),
];
const GEN: [Flag; 5] = [
    val("--model"),
    val("--prompt"),
    val("--max-new"),
    val("--temperature"),
    val("--top-k"),
];

fn table(parts: &[&[Flag]]) -> Vec<Flag> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// Adapter enumeration (EGL/Vulkan drivers) is not safe from several threads at once, and every
/// command lists adapters; tests run many commands in parallel, so they take turns.
#[cfg(test)]
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `args` (without the program name), writing the report to `out`.
pub fn run(args: &[String], out: &mut dyn Write) -> Result<(), String> {
    #[cfg(test)]
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let w = |e: std::io::Error| e.to_string();
    match cmd.as_str() {
        "-h" | "--help" | "help" => writeln!(out, "{USAGE}").map_err(w),
        "info" => info(
            &Args::parse(&table(&[&DEVICE, &[val("--model")]]), rest)?,
            out,
        ),
        "train" => {
            let a = Args::parse(
                &table(&[
                    &DEVICE,
                    &TRAIN,
                    &gpt2::SIZE_FLAGS,
                    &gpt2::TRAIN_FLAGS,
                    &[val("--quant"), val("--arch"), val("--epochs")],
                ]),
                rest,
            )?;
            match a.get("--arch").unwrap_or("gpt2") {
                "gpt2" => gpt2::train(&a, out, None),
                "llama" => {
                    if let Some(f) = gpt2::only_gpt2(&a) {
                        return Err(format!("{f} needs --arch gpt2"));
                    }
                    train(&a, out, false)
                }
                o => Err(format!("--arch: unknown '{o}' (gpt2, llama)")),
            }
        }
        "finetune-gguf" => {
            let a = Args::parse(
                &table(&[
                    &DEVICE,
                    &TRAIN,
                    &gpt2::TRAIN_FLAGS,
                    &[val("--model"), val("--quant"), val("--epochs")],
                ]),
                rest,
            )?;
            let model = a.need("--model")?;
            if gpt2::is_gpt2(model) {
                gpt2::train(&a, out, Some(gpt2::load(model)?))
            } else {
                if let Some(f) = gpt2::only_gpt2(&a) {
                    return Err(format!("{f} needs a gpt2 model"));
                }
                train(&a, out, true)
            }
        }
        "generate" | "run-gguf" => {
            let a = Args::parse(
                &table(&[
                    &DEVICE,
                    &GEN,
                    &gpt2::GEN_FLAGS,
                    &[val("--seed"), val("--max-new")],
                ]),
                rest,
            )?;
            if gpt2::is_gpt2(a.need("--model")?) {
                gpt2::generate(&a, out, cmd == "run-gguf")
            } else {
                generate(&a, out, cmd == "run-gguf")
            }
        }
        "prepare" => gpt2::prepare::run(
            &Args::parse(&gpt2::prepare::PREPARE_FLAGS, rest)?,
            rest,
            out,
        ),
        "tokenize" => gpt2::tokenize(
            &Args::parse(
                &[
                    val("--data"),
                    val("--tokenizer"),
                    val("--out"),
                    val("--doc-sep"),
                ],
                rest,
            )?,
            out,
        ),
        "export-ollama" => runtimes::export_ollama(
            &Args::parse(
                &[
                    val("--model"),
                    val("--name"),
                    val("--bin"),
                    val("--modelfile-only"),
                ],
                rest,
            )?,
            out,
        ),
        "export-lmstudio" => runtimes::export_lmstudio(
            &Args::parse(&[val("--model"), val("--repo"), val("--bin")], rest)?,
            out,
        ),
        "export-gguf" => export_gguf(
            &Args::parse(&[val("--model"), val("--out"), val("--quant")], rest)?,
            out,
        ),
        "eval" => {
            let a = Args::parse(
                &table(&[&[val("--model"), val("--data"), val("--ctx"), val("--json")]]),
                rest,
            )?;
            if gpt2::is_gpt2(a.need("--model")?) {
                gpt2::eval(&a, out)
            } else {
                eval(&a, out)
            }
        }
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}

/// Build the app (device) from the shared device flags.
fn app(a: &Args, work: Workload) -> Result<App, String> {
    let requested = DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?;
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    let d = decide(requested, work, &Gpu::list(), a.get("--gpu-adapter"))?;
    let mut app = App::new(d.choice, d.adapter.as_deref(), threads.max(1))?;
    if d.note.is_some() {
        app.fallback_note = d.note;
    }
    Ok(app)
}

/// Pre-flight: when the backend is a GPU, refuse a run whose buffers do not fit (see
/// [`preflight`]) before any of it is uploaded, else print the estimate. A CPU backend, or an
/// adapter missing from the list (limits unknown), passes.
pub fn gate(
    out: &mut dyn Write,
    app: &App,
    plan: &ferrum_wgpu::budget::BufferPlan,
    mem_mb: Option<u64>,
) -> Result<(), String> {
    let name = app.backend.name();
    let adapter = app.adapters.iter().find(|a| a.name == name);
    let Some(adapter) = adapter.filter(|_| app.backend.kind() == "gpu") else {
        return Ok(());
    };
    let budget = preflight::budget(mem_mb);
    preflight::check(plan, adapter, budget)?;
    say(out, preflight::report(plan, budget))
}

fn mem_mb(a: &Args) -> Result<Option<u64>, String> {
    a.get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()
}

fn say(out: &mut dyn Write, s: String) -> Result<(), String> {
    writeln!(out, "{s}").map_err(|e| e.to_string())
}

fn device_line(out: &mut dyn Write, app: &App) -> Result<(), String> {
    say(
        out,
        format!("device: {} ({})", app.backend.name(), app.backend.kind()),
    )?;
    app.fallback_note
        .as_ref()
        .map_or(Ok(()), |n| say(out, format!("note: {n}")))
}

/// Read a GGUF model file.
pub fn load(path: &str) -> Result<Llama, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let g = Gguf::parse(bytes).map_err(|e| format!("{path}: {e}"))?;
    from_gguf(&g).map_err(|e| format!("{path}: {e}"))
}

fn shape(m: &Llama) -> String {
    let c = &m.cfg;
    format!(
        "model: vocab {} ctx {} dim {} heads {}/{} layers {} hidden {}",
        c.vocab, c.ctx, c.dim, c.heads, c.kv_heads, c.layers, c.hidden
    )
}

fn info(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let app = app(a, Workload::Decode)?;
    device_line(out, &app)?;
    for ad in &app.adapters {
        say(
            out,
            format!("adapter: {} ({:?}, {})", ad.name, ad.kind, ad.backend),
        )?;
    }
    if let Some(p) = a.get("--model") {
        say(
            out,
            if gpt2::is_gpt2(p) {
                gpt2::shape_of(p)?
            } else {
                shape(&load(p)?)
            },
        )?;
    }
    Ok(())
}

fn quant(a: &Args) -> Result<Quant, String> {
    match a.get("--quant").unwrap_or("f32") {
        "f32" => Ok(Quant::F32),
        "q8_0" => Ok(Quant::Q8_0),
        "q4_0" => Ok(Quant::Q4_0),
        o => Err(format!("--quant: unknown '{o}' (f32, q8_0, q4_0)")),
    }
}

fn train(a: &Args, out: &mut dyn Write, finetune: bool) -> Result<(), String> {
    let model = if finetune {
        load(a.need("--model")?)?
    } else {
        let name = a.get("--preset").unwrap_or("tiny");
        preset(name).ok_or_else(|| format!("unknown preset '{name}' (tiny, small, medium)"))?;
        fresh_model(name, a.num("--seed", 1u64)?)?
    };
    let spec = TrainSpec {
        text: std::fs::read(a.need("--data")?).map_err(|e| format!("--data: {e}"))?,
        steps: a.num("--steps", 100)?,
        lr: a.num("--lr", 3e-3f32)?,
        batch: a.num("--batch", 4)?,
        context: a.num("--context", model.cfg.ctx.min(32))?,
        seed: a.num("--seed", 1u64)?,
    };
    spec.validate(&model.cfg)?;
    let q = quant(a)?;
    let target = a.need("--out")?;
    let app = app(a, Workload::Train)?;
    device_line(out, &app)?;
    say(out, shape(&model))?;
    gate(
        out,
        &app,
        &preflight::train_plan(&model, spec.batch, spec.context),
        mem_mb(a)?,
    )?;
    let progress = Arc::new(Mutex::new(Progress::default()));
    let trained = app.backend.train(&model, &spec, &progress)?;
    let losses = progress
        .lock()
        .map_err(|_| "progress lock poisoned")?
        .losses
        .clone();
    let (first, last) = (losses.first().copied(), losses.last().copied());
    if let (Some(f), Some(l)) = (first, last) {
        say(
            out,
            format!("steps: {} loss {f:.4} -> {l:.4}", losses.len()),
        )?;
    }
    atomic_write(std::path::Path::new(target), &to_gguf(&trained, q))
        .map_err(|e| format!("{target}: {e}"))?;
    say(out, format!("wrote {target}"))
}

fn generate(a: &Args, out: &mut dyn Write, verbose: bool) -> Result<(), String> {
    let model = load(a.need("--model")?)?;
    let prompt = a.need("--prompt")?;
    let opts = GenOpts {
        max_new: a.num("--max-new", 64)?,
        temperature: a.num("--temperature", 0.0f32)?,
        top_k: a.num("--top-k", 0)?,
        seed: a.num("--seed", 1u64)?,
    };
    let app = app(a, Workload::Decode)?;
    if verbose {
        device_line(out, &app)?;
        say(out, shape(&model))?;
    }
    gate(out, &app, &preflight::decode_buffers(&model), mem_mb(a)?)?;
    let new = app.backend.generate(&model, &encode(prompt), &opts)?;
    say(out, format!("{prompt}{}", decode(&new)?))
}

/// `export-gguf`: read a model (any supported precision) and write it again as f32, int8 or int4.
/// A GPT-2 file keeps its tokenizer; a Llama-style file is written byte-level.
fn export_gguf(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let (src, dst) = (a.need("--model")?, a.need("--out")?);
    let q = gpt2::quant(a, "int8")?;
    let bytes = if gpt2::is_gpt2(src) {
        let l = gpt2::load(src)?;
        ferrum_model::gpt2_gguf::to_gguf_with(&l.model, q, Some(&l.tok))
    } else {
        to_gguf(&load(src)?, q)
    };
    atomic_write(std::path::Path::new(dst), &bytes).map_err(|e| format!("{dst}: {e}"))?;
    say(out, format!("wrote {dst} ({} bytes)", bytes.len()))
}

/// Mean next-token loss of `text` under `m`, over non-overlapping windows of at most the context.
// ponytail: always on the CPU oracle; use GpuLlama::score for big models if eval gets slow.
pub fn mean_loss(m: &Llama, text: &[u8]) -> Result<(f64, usize), String> {
    let ids: Vec<usize> = text.iter().map(|&b| usize::from(b)).collect();
    let (mut sum, mut n) = (0.0f64, 0usize);
    for w in ids.chunks(m.cfg.ctx) {
        if w.len() < 2 {
            continue;
        }
        let logits = m.forward_logits(w)?;
        for i in 0..w.len() - 1 {
            let row = &logits[i * m.cfg.vocab..(i + 1) * m.cfg.vocab];
            let max = row.iter().copied().fold(f32::MIN, f32::max);
            let lse = row
                .iter()
                .map(|&x| f64::from(x - max).exp())
                .sum::<f64>()
                .ln();
            sum += lse + f64::from(max) - f64::from(row[w[i + 1]]);
            n += 1;
        }
    }
    if n == 0 {
        return Err("the data needs at least 2 bytes".into());
    }
    Ok((sum / n as f64, n))
}

fn eval(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let model = load(a.need("--model")?)?;
    let text = std::fs::read(a.need("--data")?).map_err(|e| format!("--data: {e}"))?;
    let (loss, n) = mean_loss(&model, &text)?;
    say(
        out,
        format!("tokens: {n} loss: {loss:.4} perplexity: {:.3}", loss.exp()),
    )
}

#[cfg(test)]
mod tests;
