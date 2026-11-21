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
mod mm_m;
pub use mm_m::*;
mod blockcache;
mod loss_and_grads_m;
use blockcache::*;
pub use loss_and_grads_m::*;
