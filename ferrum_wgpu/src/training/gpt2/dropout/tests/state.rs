//! Reproducibility, resume, and the off switch.

use super::{cfg, data};
use crate::device::open_for_test;
use crate::train::{Engine, GpuTrainer, StepConfig};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::Adam;

fn sc() -> StepConfig {
    StepConfig {
        adam: Adam::new(0.02),
        clip: 1.0,
    }
}

/// The losses of `n` steps of a fresh trainer with dropout `(rate, seed)`.
fn losses(rate: f32, seed: u64, n: usize) -> Option<Vec<f32>> {
    let g = open_for_test()?;
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 3), None).unwrap();
    tr.set_dropout(rate, seed);
    let (ids, targets) = data(&c, 4, 6, 0);
    Some(
        (0..n)
            .map(|_| tr.step(&ids, &targets, 4, &sc()).unwrap().loss)
            .collect(),
    )
}

#[test]
fn a_rate_of_zero_is_bit_identical_to_no_dropout() {
    let (Some(off), Some(plain)) = (losses(0.0, 99, 5), losses(0.0, 1, 5)) else {
        return;
    };
    assert_eq!(off, plain, "the seed is irrelevant when dropout is off");
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut never = GpuTrainer::new(&g, &Gpt2::init(c, 3), None).unwrap();
    let (ids, targets) = data(&c, 4, 6, 0);
    let want: Vec<f32> = (0..5)
        .map(|_| never.step(&ids, &targets, 4, &sc()).unwrap().loss)
        .collect();
    assert_eq!(off, want);
    assert_eq!(never.dropout_rate(), 0.0);
}

#[test]
fn the_same_seed_reproduces_the_run_and_another_seed_changes_it() {
    let (Some(a), Some(b), Some(c)) = (losses(0.3, 5, 6), losses(0.3, 5, 6), losses(0.3, 6, 6))
    else {
        return;
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(Some(a), losses(0.0, 5, 6));
}

#[test]
fn the_rate_is_clamped_to_the_maximum() {
    let Some(g) = open_for_test() else { return };
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(cfg(), 3), None).unwrap();
    tr.set_dropout(0.99, 1);
    assert_eq!(tr.dropout_rate(), ferrum_ref::dropout::MAX_RATE);
    tr.set_dropout(-1.0, 1);
    assert_eq!(tr.dropout_rate(), 0.0);
}

#[test]
fn a_resumed_run_continues_the_masks_of_the_run_it_resumes() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let (ids, targets) = data(&c, 4, 6, 0);
    let make = || {
        let mut t = GpuTrainer::new(&g, &Gpt2::init(c, 3), None).unwrap();
        t.set_dropout(0.3, 8);
        t
    };
    let mut straight = make();
    let want: Vec<f32> = (0..8)
        .map(|_| straight.step(&ids, &targets, 4, &sc()).unwrap().loss)
        .collect();
    let mut first = make();
    let mut got: Vec<f32> = (0..4)
        .map(|_| first.step(&ids, &targets, 4, &sc()).unwrap().loss)
        .collect();
    let snap = Engine::snapshot(&first).unwrap();
    let mut resumed = make();
    Engine::restore(&mut resumed, &snap).unwrap();
    got.extend((0..4).map(|_| resumed.step(&ids, &targets, 4, &sc()).unwrap().loss));
    assert_eq!(
        got, want,
        "steps 5.. use the masks of steps 5.. of the straight run"
    );
}
