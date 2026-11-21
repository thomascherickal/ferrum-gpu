use super::*;

/// The model shape: the checkpoint's, else the base model's, else the `--dim`/`--layers` sizes.
pub(super) fn model_config(
    resume: &Option<Checkpoint>,
    base: &Option<Gpt2>,
    tok: &ByteBpeTokenizer,
    sizes: &Sizes,
) -> Result<Config, String> {
    let cfg = match (resume, base) {
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
    Ok(cfg)
}

/// A resumed checkpoint must have been made with the same tokenizer.
pub(super) fn check_resume(resume: &Option<Checkpoint>, id: &str) -> Result<(), String> {
    match resume {
        Some(cp) if cp.corpus_id.as_deref().is_some_and(|c| c != id) => {
            Err("cannot resume: the checkpoint was made with a different tokenizer".into())
        }
        _ => Ok(()),
    }
}

/// The loop settings from the flags, with the steps per epoch and the total step count.
pub(super) fn train_config(
    a: &Args,
    sizes: &Sizes,
    context: usize,
    train_len: usize,
    validating: bool,
) -> Result<(TrainConfig, u64, u64), String> {
    let batch = a.num("--batch", 16usize)?;
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
        checkpoint_every: if validating {
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
    Ok((tc, spe, total))
}
