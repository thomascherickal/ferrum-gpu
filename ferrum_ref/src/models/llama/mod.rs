// Provenance: structure (config fields, pre-norm block, tied head, optional q/k/v biases)
// follows ferrum-cpu `ferrum_core/src/llm.rs` and `gguf/llama.rs` (commit 56a9d3a). The code is
// our own.
//! Llama / Qwen2 weights (plain host f32, matrices stored `[in, out]` row-major) and the CPU
//! oracle forward pass (full and KV-cached) the GPU decoder is checked against.

pub use grad::{gqa_bwd, gqa_fwd, loss_and_grads, rmsnorm_bwd, rope_bwd, swiglu_bwd};
pub use ops::{apply_rope, attend_row, rmsnorm, rope_tables, silu, swiglu, RopeScaling, RopeType};
pub use params::{param_lens, transpose, BLOCK_PARAMS};
pub use train::{adam_step, global_norm, LlamaTrainer, Moments};

use crate::matmul::matmul;
use crate::nn;
mod config;
mod grad;
#[cfg(test)]
mod grad_tests;
mod ops;
mod params;
mod train;
pub use config::*;
mod linear_m;

#[cfg(test)]
mod tests;
