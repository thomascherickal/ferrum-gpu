// Provenance: the window/epoch/step design (windows at a stride, a fresh shuffle each epoch
// drawn without replacement, step budget, warmup+cosine, checkpoint cadence) follows ferrum-cpu
// `slm/training_loop.rs` and `train_transformer/epoch.rs` (commit 56a9d3a). Our own code.
//! A backend-independent training loop, the engines that run its steps, and the checkpoint
//! shape. The loop is deterministic: epoch `e` draws its window order from `Rng(seed, e)`, so a
//! run resumed from step `s` continues exactly where an uninterrupted run would be.

#[cfg(test)]
mod dropout_tests;
mod engine;
mod run;
#[cfg(test)]
mod tests;

pub use engine::*;
pub use run::*;
