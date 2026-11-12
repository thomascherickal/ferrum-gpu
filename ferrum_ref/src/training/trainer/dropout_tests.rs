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

#[test]
fn a_rate_of_zero_is_bit_identical_to_plain_training() {
    let mut plain = trainer();
    let mut off = trainer().with_dropout(0.0, 99);
    assert_eq!(losses(&mut plain, 6), losses(&mut off, 6));
    assert_eq!(plain.weights(), off.weights());
}

#[test]
fn the_same_seed_gives_the_same_run_and_another_seed_a_different_one() {
    let run = |seed| losses(&mut trainer().with_dropout(0.3, seed), 6);
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
    assert_ne!(run(5), losses(&mut trainer(), 6), "dropout changes the run");
}

#[test]
fn a_resumed_run_continues_the_masks_of_the_run_it_resumes() {
    let mut straight = trainer().with_dropout(0.3, 8);
    let want = losses(&mut straight, 8);

    let mut first = trainer().with_dropout(0.3, 8);
    let mut got = losses(&mut first, 4);
    let snap = first.snapshot().unwrap();
    let mut resumed = CpuTrainer::new(Gpt2::init(cfg(), 123), 1).with_dropout(0.3, 8);
    resumed.restore(&snap).unwrap();
    got.extend(losses(&mut resumed, 4));

    assert_eq!(
        got, want,
        "step 5 onward uses steps 5.. of the same mask sequence"
    );
    assert_eq!(resumed.weights(), straight.weights());
}

#[test]
fn a_model_trained_with_dropout_still_learns_the_pattern() {
    let toks = corpus(400);
    let tc = TrainConfig {
        context: 8,
        batch: 8,
        epochs: 40,
        lr: 0.02,
        clip: 1.0,
        ..TrainConfig::default()
    };
    let mut e = trainer().with_dropout(0.2, 4);
    let r = train(&mut e, &toks, &tc, None, None, &mut NoHooks).unwrap();
    let (first, last) = (r.losses[0], *r.losses.last().unwrap());
    assert!(last < first * 0.4, "{first} -> {last}");
}

#[test]
fn dropout_and_a_tied_head_compose() {
    use crate::grad::transpose;
    let c = cfg();
    let mut e = trainer().with_tie(true).with_dropout(0.25, 1);
    let ls = losses(&mut e, 12);
    assert!(ls.last().unwrap() < &ls[0], "{ls:?}");
    let w = e.weights();
    assert_eq!(w.head, transpose(&w.tok, c.vocab, c.dim), "still tied");
}
