//! `CpuTrainer::with_dropout`: off is exactly plain training, a seed fixes the run, a resume
//! continues the masks, and a dropout-trained model still learns.
use super::*;
use crate::gpt2::{Config, Gpt2};
use crate::optim::Adam;

fn cfg() -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    }
}

fn corpus(n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8])
        .collect()
}

/// One fixed batch of 2 sequences of 8 tokens.
fn batch() -> (Vec<usize>, Vec<usize>) {
    let t = corpus(17);
    (t[..16].to_vec(), t[1..17].to_vec())
}

fn step_cfg() -> StepConfig {
    StepConfig {
        adam: Adam::new(0.02),
        clip: 1.0,
    }
}

/// The losses of `n` steps on the fixed batch.
fn losses(e: &mut CpuTrainer, n: usize) -> Vec<f32> {
    let (ids, targets) = batch();
    (0..n)
        .map(|_| e.step(&ids, &targets, 2, &step_cfg()).unwrap().loss)
        .collect()
}

fn trainer() -> CpuTrainer {
    CpuTrainer::new(Gpt2::init(cfg(), 3), 1)
}

mod a_rate_of_zero_is_bit_identical_to_plain_training;
