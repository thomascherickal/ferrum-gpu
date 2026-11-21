use super::*;

/// `train`, or `finetune-gguf` when `base` is a loaded model (its tokenizer and weights are used).
pub fn train(a: &Args, out: &mut dyn Write, base: Option<Loaded>) -> Result<(), String> {
    let x = extras::parse(a)?;
    let target = a.need("--out")?;
    let data_path = a.need("--data")?;
    let token_file = is_token_file(data_path)
        .then(|| TokenFile::open(data_path))
        .transpose()?;
    let text = match token_file {
        Some(_) => None,
        None => Some(
            String::from_utf8(std::fs::read(data_path).map_err(|e| format!("--data: {e}"))?)
                .map_err(|_| "--data: the corpus is not valid UTF-8".to_string())?,
        ),
    };
    let sizes = Sizes::from_flags(a)?;
    let resume = a
        .get("--resume")
        .map(|p| {
            std::fs::read(p)
                .map_err(|e| format!("{p}: {e}"))
                .and_then(|b| from_bytes(b).map_err(|e| format!("{p}: {e}")))
        })
        .transpose()?;
    // a corpus is text or a token file, never neither (checked above): `text` is `Some` exactly when
    // there is no token file
    let corpus_text = || text.as_deref().expect("a corpus is text or a token file");
    let (tok, base_model) = match (base, &token_file) {
        (Some(l), _) => (l.tok, Some(l.model)),
        (None, Some(tf)) => (token_file_tokenizer(a, tf, out)?, None),
        (None, None) => (tokenizer(a, corpus_text(), sizes.vocab, out)?, None),
    };
    if let Some(tf) = &token_file {
        if tf.vocab_size() != tok.vocab_size() {
            return Err(format!(
                "the token file was made for a vocabulary of {} but the tokenizer has {}",
                tf.vocab_size(),
                tok.vocab_size()
            ));
        }
    }
    let source: Box<dyn TokenSource> = match token_file {
        Some(tf) => Box::new(tf),
        None => Box::new(tok.encode_docs(corpus_text())),
    };
    let mut train_len = source.len();
    let mut val = None;
    if let Some(f) = a.get("--val") {
        let f: f64 = f
            .parse()
            .map_err(|_| format!("--val: '{f}' is not a number"))?;
        if !(0.0..0.9).contains(&f) || f == 0.0 {
            return Err("--val must be in (0, 0.9)".into());
        }
        let cut = train_len - ((train_len as f64 * f) as usize).max(2);
        val = Some(Validation {
            tokens: source.window(cut, train_len - cut)?,
            patience: a.num("--patience", 0)?,
        });
        train_len = cut;
    }
    let train_src = Prefix {
        inner: &*source,
        len: train_len,
    };
    let cfg = match (&resume, &base_model) {
        (Some(cp), _) => cp.snapshot.weights.cfg,
        (None, Some(m)) => m.cfg,
        (None, None) => Config {
            vocab: tok.vocab_size(),
            ctx: sizes.ctx,
            dim: sizes.dim,
            heads: sizes.heads,
            layers: sizes.layers,
            hidden: sizes.hidden,
        },
    };
    cfg.validate()?;
    if cfg.vocab != tok.vocab_size() {
        return Err(format!(
            "the model has {} tokens but the tokenizer has {}",
            cfg.vocab,
            tok.vocab_size()
        ));
    }
    let id = corpus_id(&tok);
    if let Some(cp) = &resume {
        if cp.corpus_id.as_deref().is_some_and(|c| c != id) {
            return Err("cannot resume: the checkpoint was made with a different tokenizer".into());
        }
    }
    let batch = a.num("--batch", 16usize)?;
    let context = cfg.ctx;
    let spe = steps_per_epoch(
        window_starts(train_len, context, a.num("--stride", 0)?).len(),
        batch,
    );
    let max_steps = match (a.get("--steps"), a.num("--tokens", 0.0f64)?) {
        (Some(_), _) => a.num("--steps", 0u64)?,
        (None, t) if t > 0.0 => (t / (batch * context) as f64).ceil() as u64,
        _ => 0,
    };
    let tc = TrainConfig {
        context,
        batch,
        stride: a.num("--stride", 0)?,
        epochs: a.num("--epochs", 100)?,
        max_steps,
        lr: a.num("--lr", sizes.lr)?,
        weight_decay: a.num("--weight_decay", 0.0)?,
        clip: a.num("--clip", 0.0)?,
        warmup: a.num("--warmup", 0)?,
        cosine: a.has("--cosine"),
        seed: a.num("--seed", 1337)?,
        checkpoint_every: if val.is_some() {
            spe
        } else {
            a.num("--checkpoint-every", 0)?
        },
    };
    let total = if max_steps > 0 {
        max_steps
    } else {
        tc.epochs as u64 * spe
    };
    let q = quant(a, "int8")?;
    let app = extras::app_for(a, &x, out)?;
    device_line(out, &app)?;
    say(
        out,
        format!(
            "model: vocab {} ctx {} dim {} heads {} layers {} hidden {}",
            cfg.vocab, cfg.ctx, cfg.dim, cfg.heads, cfg.layers, cfg.hidden
        ),
    )?;
    say(
        out,
        format!(
            "corpus: {} tokens, {} windows, {total} steps of {batch}",
            train_len,
            window_starts(train_len, context, tc.stride).len()
        ),
    )?;
    let mut plan = ferrum_wgpu::train::static_plan(&cfg);
    for (n, b) in ferrum_wgpu::train::activation_plan(&cfg, batch, context).rows() {
        plan.add(n, *b);
    }
    gate(out, &app, &plan, mem_mb(a)?)?;
    let start = resume.as_ref().map_or_else(
        || Gpt2::init(cfg, tc.seed),
        |cp| cp.snapshot.weights.clone(),
    );
    let qat = !a.has("--no-qat");
    let mut engine = extras::engine(&app, &start, qat, &x, tc.seed)?;
    if let Some(cp) = &resume {
        engine.restore(&cp.snapshot)?;
        say(out, format!("resumed at step {}", cp.snapshot.steps))?;
    }
    let csv = a
        .get("--metrics-csv")
        .map(|p| {
            let mut f = std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?;
            f.write_all(b"step,epoch,loss,lr,grad_norm,tokens_per_sec,elapsed\n")
                .map_err(|e| format!("{p}: {e}"))?;
            Ok::<_, String>(Box::new(f) as Box<dyn std::io::Write>)
        })
        .transpose()?;
    let mut hook = Run {
        out,
        verbose: a.has("--verbose"),
        csv,
        started: Instant::now(),
        max_seconds: a
            .get("--max-seconds")
            .map(|_| a.num("--max-seconds", 0.0f64))
            .transpose()?,
        stop_file: Path::new(target).with_extension("stop"),
        checkpoint: a.get("--checkpoint").map(Into::into),
        sched: tc.cosine.then_some(Schedule {
            base_lr: tc.lr,
            warmup: tc.warmup,
            total,
        }),
        lr: tc.lr,
        tokens_per_step: batch * context,
        steps_per_epoch: spe,
        total,
        val,
        best: None,
        bad_epochs: 0,
        last: None,
        reason: None,
        error: None,
    };
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
        let prompt = match &text {
            Some(t) => t.chars().take(8).collect::<String>(),
            None => tok.decode(&source.window(0, 8.min(source.len()))?),
        };
        say(
            out,
            format!(
                "sample: {}",
                crate::commands::gpt2::gen::sample_text(&app, &weights, &tok, &prompt)?
            ),
        )?;
    }
    Ok(())
}
