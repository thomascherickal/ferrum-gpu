//! Tensor element encodings: sizes, decode to f32 and encode from f32 (F32, F16, Q8_0, Q4_0, Q2_K,
//! MXFP4).
use crate::blocks::{dequant, layout};
use crate::f16::{f16_to_f32, f32_to_f16};
use crate::reader::GgufError;
use crate::{
    GGML_F16, GGML_F32, GGML_I16, GGML_I32, GGML_MXFP4, GGML_Q2_K, GGML_Q4_0, GGML_Q8_0, QK,
};
mod err;
pub use err::*;
mod encode;
pub use encode::*;

#[cfg(test)]
mod tests;
