#![allow(clippy::needless_range_loop)] // index loops read clearer in element-wise finite-difference checks
use super::grad::*;
use super::ops::{apply_rope, attend_row, rmsnorm, rope_tables, swiglu, RopeScaling, RopeType};
use super::*;
use crate::matmul::rand_vec;
use crate::optim::{Adam, Schedule};
use crate::trainer::StepConfig;

fn tiny(ty: RopeType) -> Config {
    Config {
        vocab: 7,
        ctx: 5,
        dim: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 2,
        layers: 2,
        hidden: 6,
        rope_dim: 2,
        rope_base: 10000.0,
        rope_type: ty,
        eps: 1e-5,
    }
}

fn data(cfg: &Config, batch: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1) % cfg.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2) % cfg.vocab).collect(),
    )
}

/// Central differences of `f` at every index of `x` against the analytic gradient `g`.
fn check_fd(name: &str, x: &[f32], g: &[f32], f: &dyn Fn(&[f32]) -> f32) {
    let h = 1e-2;
    for i in 0..x.len() {
        let (mut p, mut q) = (x.to_vec(), x.to_vec());
        p[i] += h;
        q[i] -= h;
        let num = (f(&p) - f(&q)) / (2.0 * h);
        let close = (g[i] - num).abs() < 2e-3 + 1e-2 * num.abs();
        assert!(close, "{name}[{i}]: analytic {} vs numeric {num}", g[i]);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn variants() -> Vec<(&'static str, Llama)> {
    let mut mqa = tiny(RopeType::Norm);
    mqa.kv_heads = 1;
    mqa.head_dim = 4;
    mqa.rope_dim = 2; // partial rotary
    let mut scaled = Llama::random(tiny(RopeType::Norm), 25, false, false);
    scaled.rope = RopeScaling {
        freq_factors: vec![2.0],
        linear_factor: 2.0,
    };
    vec![
        (
            "norm",
            Llama::random(tiny(RopeType::Norm), 21, false, false),
        ),
        (
            "neox+bias",
            Llama::random(tiny(RopeType::Neox), 22, true, false),
        ),
        ("mqa+partial", Llama::random(mqa, 23, true, false)),
        ("scaled", scaled),
    ]
}

mod adam_step_matches_the_hand_computed_first_update_and_clipping_caps_the_norm;
mod gqa_forward_agrees_with_the_row_oracle;
mod rmsnorm_bwd_matches_finite_differences;
mod token_embedding_gradient_sums_repeated_tokens;
mod whole_model_gradients_match_finite_differences_for_every_parameter_group;
