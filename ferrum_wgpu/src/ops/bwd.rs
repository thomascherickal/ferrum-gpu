//! Backward kernels of the GPT-2 stack. Same conventions as [`crate::ops`]: resident buffers,
//! recorded without waiting, every reduction in a fixed ascending order (no atomics).
use crate::device::{Gpu, GpuError};
use crate::runtime::groups_for;
use crate::tensor::GpuBuf;
mod err;
pub use err::*;
mod layernorm_bwd;
pub use layernorm_bwd::*;
mod token_csr;
pub use token_csr::*;

#[cfg(test)]
mod tests;
