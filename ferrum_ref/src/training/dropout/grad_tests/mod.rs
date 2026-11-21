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

mod no_dropout_is_bit_identical_to_the_plain_pass;
