// Provenance: which tensors are snapped and along which axis follows ferrum-cpu
// `train_transformer/qat.rs` and `quant.rs` (commit 56a9d3a). The snap itself is the Q8_0
// encoder/decoder of `ferrum_gguf`, so training sees exactly the values a Q8_0 export stores.
//! Quantisation-aware training support: snap weights onto their Q8_0 values.
//!
//! QAT runs the forward and backward pass on snapped copies of the weights while Adam updates
//! the full-precision masters (a straight-through estimator), so a model trained this way loses
//! little when exported as Q8_0.
use crate::gpt2::Gpt2;
use ferrum_gguf::quant::{decode, encode};
use ferrum_gguf::{GGML_Q8_0, QK};
mod eligible;
pub use eligible::*;

#[cfg(test)]
mod tests;
