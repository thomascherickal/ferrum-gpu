//! Quantisation-aware training for the GPU trainer: snapped copies of the weights feed the
//! forward and backward pass, while Adam keeps updating the full-precision masters
//! (a straight-through estimator). The snap is bit-exact with the CPU reference and with the
//! Q8_0 encoder the GGUF export uses.
use super::GpuTrainer;
use crate::budget::BufferPlan;
use crate::device::GpuError;
use crate::ops_qat::{q8_snap_cols, q8_snap_rows};
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::Config;
mod snap;
pub use snap::*;
mod part;

#[cfg(test)]
mod tests;
