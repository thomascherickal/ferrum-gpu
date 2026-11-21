use super::*;
use crate::budget::Source;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, BACKWARD};
use ferrum_ref::llama::{loss_and_grads, Config, LlamaTrainer, RopeScaling, RopeType};
use ferrum_ref::optim::Adam;

fn cfg(ty: RopeType) -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 16,
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        layers: 2,
        hidden: 24,
        rope_dim: 4,
        rope_base: 10000.0,
        rope_type: ty,
        eps: 1e-5,
    }
}

/// (name, model): llama-like, qwen-like with biases, MQA with partial rotary, scaled RoPE.
fn variants() -> Vec<(&'static str, Llama)> {
    let mut mqa = cfg(RopeType::Norm);
    (mqa.kv_heads, mqa.head_dim, mqa.rope_dim) = (1, 8, 4);
    mqa.heads = 2;
    let mut scaled = Llama::random(cfg(RopeType::Norm), 5, false, false);
    scaled.rope = RopeScaling {
        freq_factors: vec![1.0, 4.0],
        linear_factor: 2.0,
    };
    vec![
        ("norm", Llama::random(cfg(RopeType::Norm), 1, false, false)),
        (
            "neox+bias",
            Llama::random(cfg(RopeType::Neox), 2, true, false),
        ),
        ("mqa+partial", Llama::random(mqa, 3, true, false)),
        ("scaled", scaled),
    ]
}

fn data(c: &Config, batch: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1) % c.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2) % c.vocab).collect(),
    )
}

fn step_cfg(lr: f32) -> StepConfig {
    StepConfig {
        adam: Adam {
            weight_decay: 0.01,
            ..Adam::new(lr)
        },
        clip: 1.0,
    }
}

mod bad_input_and_tied_models_are_refused;
mod loss_and_every_gradient_match_the_cpu_oracle;
mod snapshot_and_restore_resume_a_run_exactly;
