//! The commands. Each writes its report to `out` and returns an error string instead of exiting,
//! so tests drive them with the CPU backend and a `Vec<u8>`.

use crate::flags::{val, Args, Flag};
use ferrum_gguf::Gguf;
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::{fresh_model, preset, GenOpts, Progress, TrainSpec};
use ferrum_gpu_gui::tokenizer::{decode, encode};
use ferrum_model::checkpoint::atomic_write;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::Llama;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// Usage text.
pub const USAGE: &str = "usage: ferrum-gpu <command> [flags]\n\
commands:\n  \
info          [--model F.gguf]\n  \
train         --data FILE --out F.gguf [--preset tiny|small|medium] [--steps N] [--lr X] [--batch N] [--context N] [--seed N] [--quant f32|q8_0|q4_0]\n  \
finetune-gguf --model F.gguf --data FILE --out G.gguf [same training flags]\n  \
generate      --model F.gguf --prompt TEXT [--max-new N] [--temperature X] [--top-k N] [--seed N]\n  \
run-gguf      same flags as generate; also prints the model's shape first\n  \
eval          --model F.gguf --data FILE\n\
device flags (info, train, finetune-gguf, generate, run-gguf): --device cpu|gpu|auto  --gpu-adapter NAME  --threads N";

const DEVICE: [Flag; 3] = [val("--device"), val("--gpu-adapter"), val("--threads")];
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

/// Run `args` (without the program name), writing the report to `out`.
pub fn run(args: &[String], out: &mut dyn Write) -> Result<(), String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let w = |e: std::io::Error| e.to_string();
    match cmd.as_str() {
        "-h" | "--help" | "help" => writeln!(out, "{USAGE}").map_err(w),
        "info" => info(
            &Args::parse(&table(&[&DEVICE, &[val("--model")]]), rest)?,
            out,
        ),
        "train" => train(
            &Args::parse(
                &table(&[&DEVICE, &TRAIN, &[val("--preset"), val("--quant")]]),
                rest,
            )?,
            out,
            false,
        ),
        "finetune-gguf" => train(
            &Args::parse(
                &table(&[&DEVICE, &TRAIN, &[val("--model"), val("--quant")]]),
                rest,
            )?,
            out,
            true,
        ),
        "generate" | "run-gguf" => generate(
            &Args::parse(&table(&[&DEVICE, &GEN, &[val("--seed")]]), rest)?,
            out,
            cmd == "run-gguf",
        ),
        "eval" => eval(
            &Args::parse(&table(&[&[val("--model"), val("--data")]]), rest)?,
            out,
        ),
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}

/// Build the app (device) from the shared device flags.
fn app(a: &Args) -> Result<App, String> {
    let choice = DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?;
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    App::new(choice, a.get("--gpu-adapter"), threads.max(1))
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
    let app = app(a)?;
    device_line(out, &app)?;
    for ad in &app.adapters {
        say(
            out,
            format!("adapter: {} ({:?}, {})", ad.name, ad.kind, ad.backend),
        )?;
    }
    if let Some(p) = a.get("--model") {
        say(out, shape(&load(p)?))?;
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
    let app = app(a)?;
    device_line(out, &app)?;
    say(out, shape(&model))?;
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
    let app = app(a)?;
    if verbose {
        device_line(out, &app)?;
        say(out, shape(&model))?;
    }
    let new = app.backend.generate(&model, &encode(prompt), &opts)?;
    say(out, format!("{prompt}{}", decode(&new)?))
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
