// Provenance: the block layouts and dequantization formulas follow ferrum-cpu
// ferrum_core/src/gguf/dequant.rs and format.rs at commit dbf6392c8883f293ac64b28a29c55e823179b257
// (2026-11-11), which follow ggml. Rewritten here over byte slices; our own code from here on.
//! Block formats that are dequantized to f32 on load: Q4_1, Q5_0, Q5_1, Q8_1, MXFP4 and the
//! k-quants Q2_K to Q6_K. We write only MXFP4 and Q2_K (see `mxfp4` and `q2k`); the rest are
//! read-only.
use crate::f16::f16_to_f32;
use crate::{
    GGML_MXFP4, GGML_Q2_K, GGML_Q3_K, GGML_Q4_1, GGML_Q4_K, GGML_Q5_0, GGML_Q5_1, GGML_Q5_K,
    GGML_Q6_K, GGML_Q8_1, QK, QK_K,
};
mod q4_1_block;
pub use q4_1_block::*;
mod q5_k_m;
use q5_k_m::*;

#[cfg(test)]
mod tests;
