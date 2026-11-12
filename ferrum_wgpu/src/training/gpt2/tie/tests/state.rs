//! When the head is re-derived, and what a tied head leaves untouched.

use super::{cfg, data, is_tied};
use crate::device::open_for_test;
use crate::train::{Engine, GpuTrainer, StepConfig};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::Adam;

fn sc() -> StepConfig {
    StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    }
}

#[test]
fn switching_tying_on_ties_the_head_at_once() {
    let Some(g) = open_for_test() else { return };
    let start = Gpt2::init(cfg(), 7);
    assert!(!is_tied(&start), "a fresh init has its own head");
    let mut tr = GpuTrainer::new(&g, &start, None).unwrap();
    assert!(!tr.tie_enabled());
    tr.set_tie(true).unwrap();
    assert!(tr.tie_enabled());
    assert!(is_tied(&tr.weights_host().unwrap()));
}

#[test]
fn the_head_stays_tied_after_every_step() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    tr.set_tie(true).unwrap();
    for s in 0..4 {
        let (ids, targets) = data(&c, 2, 5, s);
        tr.step(&ids, &targets, 2, &sc()).unwrap();
        assert!(is_tied(&tr.weights_host().unwrap()), "after step {}", s + 1);
    }
}

#[test]
fn the_tied_head_has_no_adam_moments() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    tr.set_tie(true).unwrap();
    for s in 0..3 {
        let (ids, targets) = data(&c, 2, 5, s);
        tr.step(&ids, &targets, 2, &sc()).unwrap();
    }
    let (m, v) = tr.moments_host().unwrap();
    assert!(m.head.iter().chain(&v.head).all(|&x| x == 0.0));
    assert!(m.tok.iter().any(|&x| x != 0.0), "the embedding is trained");
}

#[test]
fn loading_an_untied_snapshot_into_a_tied_trainer_ties_it() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let untied = Gpt2::init(c, 9);
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 1), None).unwrap();
    tr.set_tie(true).unwrap();
    tr.load(&untied, None, 0).unwrap();
    let w = tr.weights_host().unwrap();
    assert_eq!(w.tok, untied.tok, "the embedding is the snapshot's");
    assert!(is_tied(&w));
}

#[test]
fn snapshot_and_restore_through_the_engine_interface_keep_the_tie() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut a = GpuTrainer::new(&g, &Gpt2::init(c, 5), None).unwrap();
    a.set_tie(true).unwrap();
    let (ids, targets) = data(&c, 2, 5, 0);
    a.step(&ids, &targets, 2, &sc()).unwrap();
    let snap = Engine::snapshot(&a).unwrap();
    let mut b = GpuTrainer::new(&g, &Gpt2::init(c, 99), None).unwrap();
    b.set_tie(true).unwrap();
    Engine::restore(&mut b, &snap).unwrap();
    assert_eq!(b.weights_host().unwrap(), snap.weights);
    // and the two continue identically
    let next_a = a.step(&ids, &targets, 2, &sc()).unwrap().loss;
    let next_b = b.step(&ids, &targets, 2, &sc()).unwrap().loss;
    assert_eq!(next_a, next_b);
}

#[test]
fn untying_stops_re_deriving_the_head() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    tr.set_tie(true).unwrap();
    tr.set_tie(false).unwrap();
    assert!(!tr.tie_enabled());
    let (ids, targets) = data(&c, 2, 5, 0);
    tr.step(&ids, &targets, 2, &sc()).unwrap();
    assert!(
        !is_tied(&tr.weights_host().unwrap()),
        "the head trains on its own again"
    );
}
