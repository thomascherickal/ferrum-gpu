// Provenance: semantics copied from ferrum-cpu (commit 56a9d3a): `activation.rs` (gelu),
// `train_transformer/kernels.rs` (ln_fwd, attn_fwd), `loss.rs` (softmax_cross_entropy,
// clamp_prob). Rewritten over plain slices; ours from here on.
//! Forward ops of the GPT-2 stack on row-major f32 slices: the parity oracle for the GPU kernels.

mod ln_eps;
pub use ln_eps::*;
mod clamp_prob;
pub use clamp_prob::*;

#[cfg(test)]
mod tests;
