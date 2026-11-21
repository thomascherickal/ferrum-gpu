use super::*;

pub(crate) fn train(a: &Args, out: &mut dyn Write, finetune: bool) -> Result<(), String> {
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
    let q = gpt2::quant(a, "f32")?;
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
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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

pub(crate) fn generate(a: &Args, out: &mut dyn Write, verbose: bool) -> Result<(), String> {
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
    gate(
        out,
        &app,
        &preflight::decode_buffers_for(&model, &app),
        mem_mb(a)?,
    )?;
    let new = app.backend.generate(&model, &encode(prompt), &opts)?;
    say(out, format!("{prompt}{}", decode(&new)?))
}

/// `export-gguf`: read a model (any supported precision) and write it again as f32, int8 or int4.
/// A GPT-2 file keeps its tokenizer; a Llama-style file is written byte-level.
pub(crate) fn export_gguf(a: &Args, out: &mut dyn Write) -> Result<(), String> {
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
