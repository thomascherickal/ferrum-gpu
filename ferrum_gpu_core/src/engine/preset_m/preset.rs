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
