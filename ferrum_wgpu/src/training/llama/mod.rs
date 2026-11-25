//! Llama / Qwen2 fine-tuning on the GPU (f32 weights, AdamW). Weights, gradients, Adam moments
//! and every activation stay resident; one step is recorded as a single command batch and only
//! the loss and the gradient norm come back to the host. The CPU oracle is
//! `ferrum_ref::llama::LlamaTrainer`.
//! ponytail: no micro-batching, QAT, weight tying or dropout, and one batch shape's activations
//! are kept at a time (reallocated when `(batch, t)` changes); the shared `Engine` trait and the
//! training loop are still GPT-2 shaped (`Snapshot` holds a `Gpt2`). Add them when a run needs
//! them.

use crate::budget::Budget;
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::tensor::GpuBuf;
use crate::training::DeviceParams;
use acts::Acts;
use ferrum_ref::llama::Llama;
use ferrum_ref::trainer::{StepConfig, StepStats};

pub use plan::{activation_plan, param_sizes, static_plan, static_plan_for};
mod acts;
mod err;
mod plan;
mod record;
pub use err::*;
mod construct;
mod host_snapshot;
mod loss_and_step;
mod validate;

#[cfg(test)]
mod tests;
