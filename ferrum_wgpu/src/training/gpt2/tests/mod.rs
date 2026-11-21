use super::*;
use crate::budget::Source;
use crate::device::open_for_test;
use crate::tolerance::max_rel_err;
use ferrum_ref::grad::loss_and_grads;
use ferrum_ref::optim::{adam_step, clip_scale, global_norm, Adam, AdamState};
use ferrum_ref::trainer::{train, Checkpoint, CpuTrainer, Hooks, NoHooks, TrainConfig};

fn cfg() -> Config {
    Config {
        vocab: 13,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 2,
        hidden: 16,
    }
}

fn batch_data(c: &Config, batch: usize, t: usize, salt: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1 + salt) % c.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2 + salt) % c.vocab).collect(),
    )
}

fn max_err_models(a: &Gpt2, b: &Gpt2) -> f32 {
    a.params()
        .iter()
        .zip(b.params())
        .map(|(x, y)| max_rel_err(x, y))
        .fold(0.0, f32::max)
}

/// A model wide enough for Q8_0 (every width a multiple of 32).
fn qcfg() -> Config {
    Config {
        vocab: 16,
        ctx: 8,
        dim: 32,
        heads: 2,
        layers: 1,
        hidden: 64,
    }
}

fn qloop() -> TrainConfig {
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
fn pattern(n: usize, vocab: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8] % vocab)
        .collect()
}

fn loop_cfg() -> TrainConfig {
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

struct Keep(Option<Checkpoint>, Option<u64>);
impl Hooks for Keep {
    fn on_step(&mut self, step: u64, _: &StepStats) -> bool {
        self.1 != Some(step)
    }
    fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
        self.0 = Some(cp.clone());
        Ok(())
    }
}

mod a_fault_at_any_command_of_a_qat_step_applies_nothing;
mod batch_shapes_can_change_between_steps;
mod load_restores_weights_moments_and_the_step_counter_exactly;
mod micro_batches_give_the_same_update_as_the_whole_batch;
mod plans_are_consistent_with_the_allocated_buffers;
mod split_steps_are_deterministic;
