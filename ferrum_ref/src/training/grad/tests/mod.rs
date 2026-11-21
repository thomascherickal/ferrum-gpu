use super::*;
use crate::gpt2::Config;
use crate::matmul::rand_vec;

fn tiny() -> Config {
    Config {
        vocab: 7,
        ctx: 5,
        dim: 4,
        heads: 2,
        layers: 2,
        hidden: 6,
    }
}

fn data(cfg: &Config, batch: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1) % cfg.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2) % cfg.vocab).collect(),
    )
}

mod gelu_grad_matches_finite_differences_and_saturates;
mod linear_bwd_matches_finite_differences;
mod thread_count_does_not_change_the_result;
mod whole_model_gradients_match_finite_differences_for_every_parameter_group;
