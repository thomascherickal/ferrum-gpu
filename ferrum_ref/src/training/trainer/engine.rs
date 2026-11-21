// Provenance: the window/epoch/step design (windows at a stride, a fresh shuffle each epoch
// drawn without replacement, step budget, warmup+cosine, checkpoint cadence) follows ferrum-cpu
// `slm/training_loop.rs` and `train_transformer/epoch.rs` (commit 56a9d3a). Our own code.
//! The step interface ([`Engine`]), what a step needs and reports, and the CPU engine.
use crate::dropout::{step_seed, Dropout};
use crate::gpt2::Gpt2;
use crate::grad::loss_and_grads_with;
use crate::optim::{adam_step, clip_scale, global_norm, Adam, AdamState};
mod stepconfig;
pub use stepconfig::*;
mod cpu_trainer;
mod engine_trait;
