//! Llama / Qwen2 inference on the GPU with every weight, the KV cache and all activations
//! resident. Matrices are f32 or Q8_0/Q4_0 packed (dequantised in the kernel). Per token the host
//! sends one index and receives one logits row. The CPU oracle is `ferrum_ref::llama`.
//! ponytail: single-row decode uses the plain f32 `gemv` or the one-invocation-per-output
//! quantised kernel; split K across the workgroup (as `gemv.wgsl` does) if a profile on a large
//! model says decode is latency-bound.

use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops;
use crate::ops_llama;
use crate::ops_quant::{
    dequantise_bytes, effective, eligible_for, packed_words, quantise_matrix, QKind,
};
use crate::tensor::GpuBuf;
use embed::Embedding;
use ferrum_ref::llama::{rope_tables, transpose, Config, Llama};
use lin::Lin;
mod embed;
mod err_m;
mod forward;
mod lin;
#[cfg(test)]
mod quant_ext_tests;
mod shard;
#[cfg(test)]
mod sharded_tests;
pub use err_m::*;
mod dequantised_m;
pub use dequantised_m::*;
mod part;

#[cfg(test)]
mod tests;
