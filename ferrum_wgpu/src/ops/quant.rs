//! Quantised linear layers: Q8_0 and Q4_0 weights stay packed on the device and are dequantised
//! in the kernel's inner loop, which cuts the memory traffic of memory-bound decode by 3.6x
//! (Q8_0) and 6.4x (Q4_0) against f32. The dequantised values are bit-identical to
//! `ferrum_gguf::quant::decode`; only the order of the final sum differs from the CPU.
//!
//! Device layout (our own, 4-byte aligned; the GGUF block is 34 or 18 bytes): block-major
//! records, record `b * n + col` for block `b` of output column `col`, holding the f32 bits of
//! the block's scale (converted from the file's f16 on the host, exactly) and then the quants.

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;
use ferrum_gguf::f16::f16_to_f32;
use ferrum_gguf::fp8::{self, Fp8};
use ferrum_gguf::mxfp4;
use ferrum_gguf::quant::{decode, encode};
use ferrum_gguf::{GGML_F16, GGML_MXFP4, GGML_Q2_K, GGML_Q4_0, GGML_Q8_0, QK, QK_K};
use ferrum_ref::llama::transpose;
#[cfg(test)]
mod ext_tests;
mod qkind;
pub use qkind::*;
mod pack_m;
pub use pack_m::*;
mod choose_m;
pub use choose_m::*;

#[cfg(test)]
mod tests;
