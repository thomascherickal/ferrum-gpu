// Provenance: tensor names, the QKV stacking and the transposition convention follow
// ferrum-cpu `ferrum_core/src/loader/gpt2.rs` and `loader/encode.rs` (commit 56a9d3a), which
// write llama.cpp's `gpt2` layout. The code is our own.
//! `gpt2` GGUF <-> [`Gpt2`].
//!
//! GGUF stores each matrix with GGML dim 0 (contiguous) = the input dimension, so a stored
//! matrix is `[out, in]` row-major; our weights are `[in, out]`, hence a transpose each way.
//! `attn_qkv.weight` stacks the Q, K, V matrices (each transposed) along the output axis.
//! A missing `output.weight` means the head is tied to `token_embd.weight`.
use ferrum_gguf::quant::encode;
use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value};
use ferrum_ref::gpt2::{Block, Config, Gpt2};
use ferrum_ref::nn::LN_EPS;

pub use super::quant::Quant;
mod err;
pub use err::*;
mod from_gguf;
pub use from_gguf::*;
mod to_gguf;
pub use to_gguf::*;
mod ferrum_metadata;
pub use ferrum_metadata::*;

#[cfg(test)]
mod tests;
