use super::params::transpose;
use super::*;

pub fn cfg() -> Config {
    Config {
        vocab: 13,
        ctx: 12,
        dim: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        layers: 2,
        hidden: 16,
        rope_dim: 4,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

mod a_prefix_gives_the_same_logits_as_the_prefix_of_a_longer_run;
mod config_validation_names_each_problem;
