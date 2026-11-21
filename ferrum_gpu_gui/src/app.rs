//! Shared application state: the chosen backend, the current model and the training job.
use crate::engine::validation::{mean_loss, split};
use crate::engine::{preset, Backend, Progress, TrainSpec};
pub use ferrum_gpu_core::session::{DeviceChoice, Session};
use ferrum_ref::llama::Llama;
use ferrum_wgpu::device::select::AdapterSummary;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
mod jobstate;
pub use jobstate::*;
mod constructors;
mod training;
