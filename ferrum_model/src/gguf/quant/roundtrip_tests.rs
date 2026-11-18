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

#[test]
fn gpt2_files_have_the_promised_types_and_read_back_close() {
    let m = gpt();
    for q in ALL {
        let g = Gguf::parse(gpt2_gguf::to_gguf(&m, q)).unwrap();
        let want = q.tensor_type(256);
        for name in [
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert_eq!(g.tensor(name).unwrap().ggml_type, want, "{q:?} {name}");
        }
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
            0,
            "norms stay f32"
        );
        assert_eq!(g.string("ferrum.quant").unwrap(), q.tag());
        let back = gpt2_gguf::from_gguf(&g).unwrap();
        let (a, b) = (&m.blocks[0].w1, &back.blocks[0].w1);
        assert!(
            worst_share(a, b) <= bound(q),
            "{q:?}: {}",
            worst_share(a, b)
        );
    }
}

#[test]
fn llama_files_have_the_promised_types_and_read_back_close() {
    let m = llama();
    for q in ALL {
        let g = Gguf::parse(llama_gguf::to_gguf(&m, q)).unwrap();
        let want = q.tensor_type(256);
        for name in [
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "output.weight",
        ] {
            assert_eq!(g.tensor(name).unwrap().ggml_type, want, "{q:?} {name}");
        }
        // ffn_down has 512 inputs: a whole number of Q2_K blocks too
        assert_eq!(
            g.tensor("blk.0.ffn_down.weight").unwrap().ggml_type,
            q.tensor_type(512)
        );
        let back = llama_gguf::from_gguf(&g).unwrap();
        let (a, b) = (&m.blocks[0].gate, &back.blocks[0].gate);
        assert!(
            worst_share(a, b) <= bound(q),
            "{q:?}: {}",
            worst_share(a, b)
        );
    }
}

#[test]
fn a_matrix_that_is_not_whole_blocks_falls_back() {
    // width 96: whole blocks of 32 but not of 256
    let m = Gpt2::init(
        GptConfig {
            vocab: 20,
            ctx: 8,
            dim: 96,
            heads: 4,
            layers: 1,
            hidden: 192,
        },
        3,
    );
    for (q, want) in [
        (Quant::Q2K, ferrum_gguf::GGML_Q4_0),
        (Quant::Fp4, ferrum_gguf::GGML_MXFP4),
        (Quant::F16, ferrum_gguf::GGML_F16),
    ] {
        let g = Gguf::parse(gpt2_gguf::to_gguf(&m, q)).unwrap();
        assert_eq!(
            g.tensor("blk.0.ffn_up.weight").unwrap().ggml_type,
            want,
            "{q:?}"
        );
    }
}
