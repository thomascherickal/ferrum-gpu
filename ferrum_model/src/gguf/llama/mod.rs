// Provenance: tensor names, metadata keys, the `[in, out]` shape checks, the tied-head rule, the
// rope-scaling policy (none / linear / `rope_freqs.weight`, everything else refused) and the
// arch -> RoPE pairing rule (llama = NORM, qwen2 = NEOX) follow ferrum-cpu
// `ferrum_core/src/gguf/llama.rs` (commit 56a9d3a), which follows llama.cpp. The code is our own.
//! `llama` / `qwen2` GGUF <-> [`Llama`].
//!
//! GGUF stores each matrix with GGML dim 0 (contiguous) = the input dimension, so a stored
//! matrix is `[out, in]` row-major; our weights are `[in, out]`, hence a transpose each way. A
//! missing `output.weight` means the head is tied to `token_embd.weight`. Biases are accepted on
//! `attn_q`, `attn_k`, `attn_v` only; a bias anywhere else is refused rather than dropped.

use crate::gpt2_gguf::Quant;
use ferrum_gguf::quant::encode;
use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value};
use ferrum_ref::llama::{transpose, Block, Config, Llama, RopeScaling, RopeType};
mod err_m;
mod from_gguf_m;
pub use from_gguf_m::*;
mod metadata_m;
use err_m::*;
pub use metadata_m::*;

#[cfg(test)]
mod tests;
