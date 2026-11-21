// Provenance: AdamW update order, decoupled decay and global-norm clipping are the ones in
// `crate::optim` (copied there from ferrum-cpu commit 56a9d3a); duplicated over [`Llama`] rather
// than generalised because `Gpt2` is concrete in the optimizer and the loop.
// ponytail: duplicated; make `optim` generic over a `params()` trait when a third model appears.
//! AdamW and a single-step trainer for [`Llama`] on the CPU: the oracle for the GPU trainer.
use super::grad::loss_and_grads;
use super::Llama;
use crate::optim::{clip_scale, Adam};
use crate::trainer::{StepConfig, StepStats};
mod moments;
pub use moments::*;
mod part;
