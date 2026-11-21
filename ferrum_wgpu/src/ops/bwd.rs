//! Backward kernels of the GPT-2 stack. Same conventions as [`crate::ops`]: resident buffers,
//! recorded without waiting, every reduction in a fixed ascending order (no atomics).
use crate::device::{Gpu, GpuError};
use crate::runtime::groups_for;
use crate::tensor::GpuBuf;
mod err_m;
pub use err_m::*;
mod token_csr_m;
pub use token_csr_m::*;

#[cfg(test)]
mod tests;
