//! GPT-2 inference on the GPU with every weight, the KV cache and all activations resident.
//! Per token the host sends one index and receives one logits row.
//! ponytail: each dispatch is its own submit (about 0.03 ms on the MX330); batch the whole
//! token into one command encoder if a profile shows submit cost dominating.
use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops;
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::{Block, Config, Gpt2};
mod err_m;
pub use err_m::*;
mod part;

#[cfg(test)]
mod tests;
