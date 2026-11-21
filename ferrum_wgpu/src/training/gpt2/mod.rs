//! GPT-2 training on the GPU. Weights, gradients, Adam moments and every activation stay
//! resident; one step is recorded as a single command batch, and only the loss and the
//! gradient norm come back to the host.
//! ponytail: one batch shape's activations are kept at a time (reallocated when `(batch, t)`
//! changes). Tied heads, QAT and FFN dropout are in; each is its own module.

use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops_opt::{accumulate, adam_update, norm_scale, parts_for, sumsq_parts};
use crate::tensor::GpuBuf;
use acts::Acts;
use ferrum_ref::gpt2::{Config, Gpt2};
use std::cell::Cell;

pub use plan::{accum_plan, activation_plan, static_plan};
pub use qat::qat_plan;

pub use ferrum_ref::trainer::{Engine, Snapshot, StepConfig, StepStats};
mod acts;
mod dropout;
mod engine;
mod err;
mod plan;
mod qat;
mod record;
mod tie;
pub use err::*;
mod construct;
mod load;
mod loss_and_step;
mod micro_batch;
mod optimize;
mod split_step;

#[cfg(test)]
mod tests;
