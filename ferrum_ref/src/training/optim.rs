// Provenance: Adam/AdamW update order and decoupled-decay rule, global-norm clipping and the
// warmup+cosine schedule follow ferrum-cpu `optim.rs` and `train_transformer/param.rs`
// (commit 56a9d3a); rewritten over plain slices, ours from here on.
//! AdamW, gradient clipping and the learning-rate schedule on the CPU (the oracle for the GPU
//! optimizer kernels).
use crate::gpt2::Gpt2;
mod adam;
pub use adam::*;
mod clip_grad_norm;
pub use clip_grad_norm::*;

#[cfg(test)]
mod tests;
