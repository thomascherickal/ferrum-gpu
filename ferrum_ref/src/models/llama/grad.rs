// Provenance: the gradient formulas are the standard ones for RMSNorm, RoPE (the transpose of a
// rotation), grouped-query attention and SwiGLU, checked against finite differences here; the
// block order follows ferrum-cpu `llm_train/blocks.rs` (commit 56a9d3a). The code is our own.
//! Training forward/backward for [`Llama`] on the CPU: the oracle for the GPU trainer. Batched:
//! `ids` is `batch` sequences of length `t`. Tied heads are refused (train an untied copy).
use super::ops::{apply_rope, rmsnorm, rope_tables, silu, swiglu, RopeType};
use super::{Config, Llama};
use crate::grad::linear_bwd;
use crate::matmul::MatmulError;
use crate::nn;
use crate::tuned::matmul_tuned;
mod mm;
pub use mm::*;
mod gqa_fwd;
pub use gqa_fwd::*;
mod block_backward;
mod block_cache;
mod block_pass;
mod loss_and_grads;
use block_cache::*;
use block_pass::Pass;
pub use loss_and_grads::*;
