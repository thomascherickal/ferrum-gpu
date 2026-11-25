//! Forward kernels of the Llama / Qwen stack: RMSNorm, RoPE from host-built tables, SwiGLU, an
//! embedding gather and grouped-query attention (full and KV-cached). Every function records one
//! or more dispatches on resident buffers and does not wait; read results with [`Gpu::read`].
//! The CPU oracle for each is in `ferrum_ref::llama`.

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::RopeType;
mod rmsnorm;
pub use rmsnorm::*;
mod gather_rows;
pub use gather_rows::*;
mod attention_decode_gqa;
pub use attention_decode_gqa::*;
#[cfg(test)]
pub(crate) mod tests;
