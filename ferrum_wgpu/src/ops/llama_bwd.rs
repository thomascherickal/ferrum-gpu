//! Backward kernels of the Llama / Qwen stack: RMSNorm, SwiGLU and grouped-query attention.
//! (RoPE's backward is [`crate::ops_llama::rope`] called with a negated sine table: the
//! transpose of a rotation is the rotation by the opposite angle. Linear layers reuse
//! [`crate::ops_bwd::matmul_ex`].) Same conventions as [`crate::ops_bwd`]: resident buffers,
//! recorded without waiting, every reduction in a fixed ascending order, no atomics. The CPU
//! oracle for each is in `ferrum_ref::llama`.

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;
mod rmsnorm_bwd;
pub use rmsnorm_bwd::*;
mod attention_gqa_bwd;
pub use attention_gqa_bwd::*;
#[cfg(test)]
mod tests;
