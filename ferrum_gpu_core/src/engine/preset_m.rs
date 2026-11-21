use super::*;

/// A named model size.
pub fn preset(name: &str) -> Option<Config> {
    let (ctx, dim, heads, kv, layers, hidden) = match name {
        "tiny" => (64, 32, 4, 2, 2, 64),
        "small" => (128, 64, 4, 2, 3, 176),
        "medium" => (128, 128, 8, 2, 4, 352),
        _ => return None,
    };
    let head_dim = dim / heads;
    Some(Config {
        vocab: VOCAB,
        ctx,
        dim,
        heads,
        kv_heads: kv,
        head_dim,
        layers,
        hidden,
        rope_dim: head_dim,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    })
}

/// A freshly initialised (untied, bias-free) model of a preset.
pub fn fresh_model(name: &str, seed: u64) -> Result<Llama, String> {
    fresh_model_with(name, seed, false)
}

/// Like [`fresh_model`]; `tied` shares the token embedding with the output head.
pub fn fresh_model_with(name: &str, seed: u64, tied: bool) -> Result<Llama, String> {
    let cfg =
        preset(name).ok_or_else(|| format!("unknown preset '{name}' (tiny, small, medium)"))?;
    Ok(Llama::random(cfg, seed, false, tied))
}

/// What a training run needs.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainSpec {
    /// Training text as bytes (the byte tokenizer's ids).
    pub text: Vec<u8>,
    /// Optimizer steps.
    pub steps: usize,
    /// Peak learning rate.
    pub lr: f32,
    /// Windows per step.
    pub batch: usize,
    /// Tokens per window.
    pub context: usize,
    /// Seed of the window choice.
    pub seed: u64,
}

impl TrainSpec {
    /// Reject specs no model can train on.
    pub fn validate(&self, cfg: &Config) -> Result<(), String> {
        if self.steps == 0 || self.steps > 100_000 {
            return Err("steps must be in 1..=100000".into());
        }
        if self.batch == 0 || self.batch > 256 {
            return Err("batch must be in 1..=256".into());
        }
        if self.context == 0 || self.context > cfg.ctx {
            return Err(format!("context must be in 1..={}", cfg.ctx));
        }
        if self.text.len() <= self.context {
            return Err(format!(
                "the text has {} bytes but needs more than the context of {}",
                self.text.len(),
                self.context
            ));
        }
        if !(self.lr.is_finite() && self.lr > 0.0 && self.lr <= 1.0) {
            return Err("lr must be in (0, 1]".into());
        }
        Ok(())
    }
}

/// Live state of a training run, shared with the HTTP thread.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    /// Steps finished.
    pub step: usize,
    /// Steps planned.
    pub total: usize,
    /// Mean loss of every finished step.
    pub losses: Vec<f32>,
    /// Set by the HTTP thread to end the run early.
    pub stop: bool,
    /// Loss on the held-out text before training (a run with a validation share only).
    pub val_before: Option<f32>,
    /// Loss on the held-out text after training.
    pub val_after: Option<f32>,
}

/// `batch` windows of `context` tokens and their next-token targets, starting at random offsets.
pub fn make_batch(
    text: &[u8],
    context: usize,
    batch: usize,
    rng: &mut Rng,
) -> (Vec<usize>, Vec<usize>) {
    let starts = text.len() - context; // windows need one byte past the end for the target
    let (mut ids, mut targets) = (
        Vec::with_capacity(batch * context),
        Vec::with_capacity(batch * context),
    );
    for _ in 0..batch {
        let s = rng.below(starts);
        ids.extend(text[s..s + context].iter().map(|&b| usize::from(b)));
        targets.extend(text[s + 1..s + context + 1].iter().map(|&b| usize::from(b)));
    }
    (ids, targets)
}

/// One optimizer step: `(ids, targets, batch, config)` to the step's stats.
pub type StepFn<'a> =
    dyn FnMut(&[usize], &[usize], usize, &StepConfig) -> Result<StepStats, String> + 'a;

/// The progress record, recovered if a panicking thread poisoned the lock (it is plain counters).
pub(super) fn lock(progress: &Arc<Mutex<Progress>>) -> std::sync::MutexGuard<'_, Progress> {
    progress
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
