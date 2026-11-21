//! A whole model stored as MXFP4, FP8 or Q2_K on the device runs like its CPU oracle.
use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD, Q2K_FORWARD};
use ferrum_gguf::Gguf;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::RopeType;

const IDS: [usize; 8] = [3, 16, 0, 9, 9, 4, 11, 2];

fn cfg(dim: usize, hidden: usize, head_dim: usize) -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim,
        heads: dim / head_dim,
        kv_heads: dim / head_dim / 2,
        head_dim,
        layers: 2,
        hidden,
        rope_dim: head_dim,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

fn feed_error(g: &Gpu, w: &Llama, kind: QKind) -> f32 {
    let want = dequantised(w, kind).unwrap().forward_logits(&IDS).unwrap();
    let mut m = GpuLlama::new(g, w, Some(kind), None).unwrap();
    let mut worst = 0.0f32;
    for (i, &t) in IDS.iter().enumerate() {
        worst = worst.max(max_rel_err(
            &m.feed(t).unwrap(),
            &want[i * 17..(i + 1) * 17],
        ));
    }
    worst
}

mod the_32_wide_formats_match_their_oracle_with_and_without_biases;
mod the_buffer_plan_counts_the_fallback_widths;
