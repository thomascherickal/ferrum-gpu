//! The engine behind the GUI: model presets, the shared training loop and sampling loop, and a
//! [`Backend`] trait with a CPU implementation (the oracle) and a GPU implementation. The two
//! share every line except the step and the single-token decode.

mod cpu;
mod gpu;
#[cfg(test)]
mod tests;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;

use crate::tokenizer::VOCAB;
use ferrum_model::sampling::{sample, Params, Rng};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_ref::optim::{Adam, Schedule};
pub use ferrum_ref::trainer::Engine;
use ferrum_ref::trainer::{StepConfig, StepStats};
use std::sync::{Arc, Mutex};

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
    let cfg =
        preset(name).ok_or_else(|| format!("unknown preset '{name}' (tiny, small, medium)"))?;
    Ok(Llama::random(cfg, seed, false, false))
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

/// The training loop both backends share: warmup + cosine learning rate, global-norm clip 1.0,
/// AdamW, progress after every step, early stop. `step` runs one optimizer step.
pub fn run_training(
    spec: &TrainSpec,
    progress: &Arc<Mutex<Progress>>,
    step: &mut StepFn,
) -> Result<(), String> {
    let sched = Schedule {
        base_lr: spec.lr,
        warmup: (spec.steps as u64 / 10).clamp(1, 20),
        total: spec.steps as u64,
    };
    let mut rng = Rng::new(spec.seed);
    progress.lock().map_err(|_| "progress lock poisoned")?.total = spec.steps;
    for i in 1..=spec.steps {
        if progress.lock().map_err(|_| "progress lock poisoned")?.stop {
            break;
        }
        let (ids, targets) = make_batch(&spec.text, spec.context, spec.batch, &mut rng);
        let cfg = StepConfig {
            adam: Adam {
                weight_decay: 0.01,
                ..Adam::new(sched.lr_at(i as u64))
            },
            clip: 1.0,
        };
        let stats = step(&ids, &targets, spec.batch, &cfg)?;
        if !stats.loss.is_finite() {
            return Err(format!(
                "the loss became {} at step {i}; lower the learning rate",
                stats.loss
            ));
        }
        let mut p = progress.lock().map_err(|_| "progress lock poisoned")?;
        p.step = i;
        p.losses.push(stats.loss);
    }
    Ok(())
}

/// Sampling controls for a generation request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenOpts {
    /// Tokens to add after the prompt.
    pub max_new: usize,
    /// `0.0` is greedy.
    pub temperature: f32,
    /// `0` is off.
    pub top_k: usize,
    /// Seed of the sampler.
    pub seed: u64,
}

/// Feed `prompt`, then sample `opts.max_new` tokens (stopping when the context is full), using
/// `feed` to turn a token into the next-token logits. Returns only the new tokens.
pub fn generate_with(
    feed: &mut dyn FnMut(usize) -> Result<Vec<f32>, String>,
    ctx: usize,
    prompt: &[usize],
    opts: &GenOpts,
) -> Result<Vec<usize>, String> {
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    if prompt.len() >= ctx {
        return Err(format!(
            "the prompt has {} tokens but the context is {ctx}",
            prompt.len()
        ));
    }
    let params = Params {
        temperature: opts.temperature,
        top_k: opts.top_k,
        ..Params::default()
    };
    let mut rng = Rng::new(opts.seed);
    let mut logits = Vec::new();
    for &t in prompt {
        logits = feed(t)?;
    }
    let mut history = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < opts.max_new && history.len() < ctx {
        let next = sample(&logits, &params, &history, &mut rng);
        out.push(next);
        history.push(next);
        if history.len() < ctx && out.len() < opts.max_new {
            logits = feed(next)?;
        }
    }
    Ok(out)
}

/// A one-token-at-a-time GPT-2 decoder: feed a token id, get the next-token logits.
pub type Gpt2Feed<'a> = Box<dyn FnMut(usize) -> Result<Vec<f32>, String> + 'a>;

/// A place to run models.
pub trait Backend: Send + Sync {
    /// A GPT-2 training engine for `w` (quantisation-aware when `qat`), borrowed from the backend.
    fn gpt2_engine<'a>(&'a self, _w: &Gpt2, _qat: bool) -> Result<Box<dyn Engine + 'a>, String> {
        Err(format!(
            "GPT-2 training is not available on {}",
            self.name()
        ))
    }

    /// A GPT-2 decoder with a fresh KV cache.
    fn gpt2_decoder<'a>(&'a self, _w: &Gpt2) -> Result<Gpt2Feed<'a>, String> {
        Err(format!(
            "GPT-2 decoding is not available on {}",
            self.name()
        ))
    }

    /// A Llama decoder with a fresh KV cache (token in, next-token logits out).
    fn llama_decoder<'a>(&'a self, _m: &'a Llama) -> Result<Gpt2Feed<'a>, String> {
        Err(format!("decoding is not available on {}", self.name()))
    }

    /// `"cpu"` or `"gpu"`.
    fn kind(&self) -> &'static str;
    /// Human-readable device name.
    fn name(&self) -> String;
    /// Train `model` on `spec.text`, updating `progress`; returns the trained weights.
    fn train(
        &self,
        model: &Llama,
        spec: &TrainSpec,
        progress: &Arc<Mutex<Progress>>,
    ) -> Result<Llama, String>;
    /// Continue `prompt` with sampled tokens.
    fn generate(
        &self,
        model: &Llama,
        prompt: &[usize],
        opts: &GenOpts,
    ) -> Result<Vec<usize>, String>;
}
