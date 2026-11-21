use super::*;
use crate::tolerance::max_rel_err;
use ferrum_ref::trainer::{Checkpoint, Hooks, TrainConfig};

pub(super) fn cfg() -> Config {
    Config {
        vocab: 13,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 2,
        hidden: 16,
    }
}

pub(super) fn batch_data(
    c: &Config,
    batch: usize,
    t: usize,
    salt: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1 + salt) % c.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2 + salt) % c.vocab).collect(),
    )
}

pub(super) fn max_err_models(a: &Gpt2, b: &Gpt2) -> f32 {
    a.params()
        .iter()
        .zip(b.params())
        .map(|(x, y)| max_rel_err(x, y))
        .fold(0.0, f32::max)
}

/// A model wide enough for Q8_0 (every width a multiple of 32).
pub(super) fn qcfg() -> Config {
    Config {
        vocab: 16,
        ctx: 8,
        dim: 32,
        heads: 2,
        layers: 1,
        hidden: 64,
    }
}

pub(super) fn qloop() -> TrainConfig {
    TrainConfig {
        context: 8,
        batch: 6,
        max_steps: 25,
        lr: 0.01,
        clip: 1.0,
        seed: 3,
        ..TrainConfig::default()
    }
}

/// A learnable corpus: a repeating pattern with a few distinct tokens.
pub(super) fn pattern(n: usize, vocab: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8] % vocab)
        .collect()
}

pub(super) fn loop_cfg() -> TrainConfig {
    TrainConfig {
        context: 8,
        batch: 6,
        max_steps: 60,
        lr: 0.02,
        weight_decay: 0.01,
        clip: 1.0,
        warmup: 5,
        cosine: true,
        seed: 11,
        ..TrainConfig::default()
    }
}

pub(super) struct Keep(pub(super) Option<Checkpoint>, pub(super) Option<u64>);
impl Hooks for Keep {
    fn on_step(&mut self, step: u64, _: &StepStats) -> bool {
        self.1 != Some(step)
    }
    fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
        self.0 = Some(cp.clone());
        Ok(())
    }
}
