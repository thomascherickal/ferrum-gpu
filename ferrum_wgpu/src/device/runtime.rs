//! Device runtime shared by every kernel: error scopes, device-health, a pipeline cache, one
//! generic dispatch path, and plan-then-allocate.
use crate::budget::{Budget, BufferPlan};
use crate::device::{block_on, Gpu, GpuError};
use crate::tensor::GpuBuf;
use std::sync::atomic::Ordering;
mod map_error;
pub use map_error::*;
mod copy_and_alloc;
mod dispatch;
mod fault_injection;
mod pipeline_cache;

#[cfg(test)]
mod tests;
