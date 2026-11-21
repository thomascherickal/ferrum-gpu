use super::*;

/// `train`, or `finetune-gguf` when `base` is a loaded model (its tokenizer and weights are used).
pub fn train(a: &Args, out: &mut dyn Write, base: Option<Loaded>) -> Result<(), String> {
    let x = extras::parse(a)?;
    let target = a.need("--out")?;
    let mut corpus = Corpus::load(a)?;
    let sizes = Sizes::from_flags(a)?;
    let resume = load_resume(a)?;
    let (tok, base_model) = match (base, &corpus.token_file) {
        (Some(l), _) => (l.tok, Some(l.model)),
        (None, Some(tf)) => (token_file_tokenizer(a, tf, out)?, None),
        (None, None) => (tokenizer(a, corpus.text(), sizes.vocab, out)?, None),
    };
    let source = corpus.source(&tok)?;
    let (train_len, val) = split_validation(a, &*source)?;
    let train_src = Prefix {
        inner: &*source,
        len: train_len,
    };
    let cfg = model_config(&resume, &base_model, &tok, &sizes)?;
    let id = corpus_id(&tok);
    check_resume(&resume, &id)?;
    let (tc, spe, total) = train_config(a, &sizes, cfg.ctx, train_len, val.is_some())?;
    let q = quant(a, "int8")?;
    let app = extras::app_for(a, &x, out)?;
    device_line(out, &app)?;
    announce(out, &cfg, train_len, &tc, total)?;
    gate_memory(a, out, &app, &cfg, &tc)?;
    let start = resume.as_ref().map_or_else(
        || Gpt2::init(cfg, tc.seed),
        |cp| cp.snapshot.weights.clone(),
    );
    let mut engine = extras::engine(&app, &start, !a.has("--no-qat"), &x, tc.seed)?;
    if let Some(cp) = &resume {
        engine.restore(&cp.snapshot)?;
        say(out, format!("resumed at step {}", cp.snapshot.steps))?;
    }
    let shape = Shape {
        tc: &tc,
        steps_per_epoch: spe,
        total,
    };
    let mut hook = run_hook(a, out, target, shape, val)?;
    let expect: Option<RunShape> = resume.as_ref().map(|cp| cp.shape.clone());
    let report = run_loop(
        &mut *engine,
        &train_src,
        &tc,
        expect.as_ref(),
        Some(&id),
        &mut hook,
    )?;
    if let Some(e) = hook.error.take() {
        return Err(e);
    }
    let weights = hook
        .best
        .take()
        .map(|b| b.1)
        .or(hook.last.take())
        .ok_or("training produced no weights")?;
    let reason = hook.reason.take();
    let out = hook.out;
    if let (Some(f), Some(l)) = (report.losses.first(), report.losses.last()) {
        say(
            out,
            format!("steps: {} loss {f:.4} -> {l:.4}", report.steps_run),
        )?;
    }
    if let Some(r) = reason {
        say(out, format!("stopped early: {r}"))?;
    }
    atomic_write(Path::new(target), &to_gguf_with(&weights, q, Some(&tok)))
        .map_err(|e| format!("{target}: {e}"))?;
    say(out, format!("wrote {target}"))?;
    if a.has("--sample") {
        let prompt = match &corpus.text {
            Some(t) => t.chars().take(8).collect::<String>(),
            None => tok.decode(&source.window(0, 8.min(source.len()))?),
        };
        let sample = crate::commands::gpt2::gen::sample_text(&app, &weights, &tok, &prompt)?;
        say(out, format!("sample: {sample}"))?;
    }
    Ok(())
}
