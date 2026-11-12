//! Dropout inside the forward/backward pass: no mask changes nothing, the gradient is the
//! gradient of the masked network, and the mask follows the seed.

use super::*;
use crate::gpt2::{Config, Gpt2};
use crate::grad::{loss_and_grads, loss_and_grads_with};

fn cfg() -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 2,
        hidden: 16,
    }
}

fn batch() -> (Vec<usize>, Vec<usize>) {
    let toks: Vec<usize> = (0..17)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8])
        .collect();
    (toks[..16].to_vec(), toks[1..17].to_vec())
}

fn run(m: &Gpt2, d: Option<Dropout>) -> (f32, Gpt2) {
    let (ids, targets) = batch();
    loss_and_grads_with(m, &ids, &targets, 2, 1, d).unwrap()
}

#[test]
fn no_dropout_is_bit_identical_to_the_plain_pass() {
    let m = Gpt2::init(cfg(), 3);
    let (ids, targets) = batch();
    let (l0, g0) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    let (l1, g1) = run(&m, None);
    assert_eq!((l0, g0), (l1, g1));
}

#[test]
fn the_same_seed_reproduces_the_pass_and_another_seed_changes_it() {
    let m = Gpt2::init(cfg(), 3);
    let d = |seed| Dropout::new(0.3, seed);
    assert_eq!(run(&m, d(5)), run(&m, d(5)));
    assert_ne!(
        run(&m, d(5)).0,
        run(&m, d(6)).0,
        "a different mask, a different loss"
    );
    assert_ne!(run(&m, d(5)).0, run(&m, None).0);
}

/// One parameter to perturb: its name, how to bump it, and its analytic gradient.
struct Probe {
    name: &'static str,
    bump: fn(&mut Gpt2, f32),
    analytic: f32,
}

/// The central difference of the loss (same dropout seed, so the same mask) along `probe`.
fn numeric(m: &Gpt2, d: Option<Dropout>, probe: &Probe, eps: f32) -> f32 {
    let at = |e: f32| {
        let mut p = m.clone();
        (probe.bump)(&mut p, e);
        run(&p, d).0
    };
    (at(eps) - at(-eps)) / (2.0 * eps)
}

#[test]
fn the_gradient_is_that_of_the_masked_network() {
    let m = Gpt2::init(cfg(), 4);
    let d = Dropout::new(0.4, 11);
    let (_, g) = run(&m, d);
    // a first-block and a last-block FFN weight, and an embedding row that is read
    let probes = [
        Probe {
            name: "blocks[0].w1[5]",
            bump: |m, e| m.blocks[0].w1[5] += e,
            analytic: g.blocks[0].w1[5],
        },
        Probe {
            name: "blocks[0].w2[9]",
            bump: |m, e| m.blocks[0].w2[9] += e,
            analytic: g.blocks[0].w2[9],
        },
        Probe {
            name: "blocks[1].w1[40]",
            bump: |m, e| m.blocks[1].w1[40] += e,
            analytic: g.blocks[1].w1[40],
        },
        Probe {
            name: "tok[3*8+2]",
            bump: |m, e| m.tok[3 * 8 + 2] += e,
            analytic: g.tok[3 * 8 + 2],
        },
    ];
    for p in &probes {
        let n = numeric(&m, d, p, 5e-3);
        assert!(
            (n - p.analytic).abs() <= 3e-3 + 0.05 * p.analytic.abs(),
            "{}: numeric {n} vs analytic {}",
            p.name,
            p.analytic
        );
    }
}

#[test]
fn a_dropped_unit_receives_no_gradient() {
    // with nearly everything dropped, most FFN input-weight gradients vanish (dropped units pass
    // nothing back through GELU), unlike the undropped network
    let m = Gpt2::init(cfg(), 4);
    let (_, plain) = run(&m, None);
    let (_, heavy) = run(&m, Dropout::new(0.95, 2));
    let zeros = |g: &Gpt2| g.blocks[0].b1.iter().filter(|&&x| x == 0.0).count();
    assert!(
        zeros(&heavy) > zeros(&plain),
        "{} vs {}",
        zeros(&heavy),
        zeros(&plain)
    );
}
