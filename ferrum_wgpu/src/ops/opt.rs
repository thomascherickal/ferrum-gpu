//! Optimizer kernels: deterministic global gradient norm, clip scale, and the AdamW update.
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::optim::Adam;
mod chunk;
pub use chunk::*;
mod accumulate;
pub use accumulate::*;

#[cfg(test)]
mod tests;
