//! Both writers, every precision: the file has the tensor types the precision promises and reads
//! back close to the model that was written.
use super::*;
use crate::{gpt2_gguf, llama_gguf};
use ferrum_gguf::Gguf;
use ferrum_ref::gpt2::{Config as GptConfig, Gpt2};
use ferrum_ref::llama::{Config, Llama, RopeType};

const ALL: [Quant; 6] = [
    Quant::F32,
    Quant::F16,
    Quant::Q8_0,
    Quant::Q4_0,
    Quant::Fp4,
    Quant::Q2K,
];

fn gpt() -> Gpt2 {
    Gpt2::init(
        GptConfig {
            vocab: 20,
            ctx: 8,
            dim: 256,
            heads: 4,
            layers: 1,
            hidden: 512,
        },
        3,
    )
}

fn llama() -> Llama {
    Llama::random(
        Config {
            vocab: 20,
            ctx: 8,
            dim: 256,
            heads: 4,
            kv_heads: 2,
            head_dim: 64,
            layers: 1,
            hidden: 512,
            rope_dim: 64,
            rope_base: 10000.0,
            rope_type: RopeType::Norm,
            eps: 1e-5,
        },
        4,
        false,
        false,
    )
}

/// Largest absolute error as a share of the largest weight, per precision (2-bit is coarse).
fn bound(q: Quant) -> f32 {
    match q {
        Quant::F32 => 0.0,
        Quant::F16 => 0.001,
        Quant::Q8_0 => 0.01,
        Quant::Q4_0 => 0.15,
        Quant::Fp4 => 0.35,
        Quant::Q2K => 1.0,
    }
}

fn worst_share(a: &[f32], b: &[f32]) -> f32 {
    let amax = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / amax
}

mod gpt2_files_have_the_promised_types_and_read_back_close;
