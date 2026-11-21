use super::*;
use crate::budget::Source;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD};
use ferrum_gguf::Gguf;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::{Cache, RopeScaling, RopeType};

fn cfg() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

/// (name, config, biases, tied): llama-like, qwen-like with biases and a tied head, partial rotary
/// with MQA, and a width that is not a multiple of the quantisation block.
fn variants() -> Vec<(&'static str, Config, bool, bool)> {
    let mut neox = cfg();
    neox.rope_type = RopeType::Neox;
    let mut mqa = cfg();
    (mqa.kv_heads, mqa.rope_dim) = (1, 4);
    let mut odd = cfg();
    (odd.dim, odd.heads, odd.kv_heads, odd.head_dim, odd.rope_dim) = (24, 3, 3, 8, 8);
    vec![
        ("llama", cfg(), false, false),
        ("qwen", neox, true, true),
        ("mqa+partial", mqa, false, false),
        ("dim24", odd, true, false),
    ]
}

const IDS: [usize; 10] = [3, 16, 0, 9, 9, 4, 11, 2, 7, 5];

mod feed_matches_the_oracle_at_every_position_of_a_full_context;
mod greedy_generation_agrees_with_the_oracle_token_for_token;
