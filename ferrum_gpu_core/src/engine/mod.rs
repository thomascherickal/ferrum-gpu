//! The engine behind the GUI: model presets, the shared training loop and sampling loop, and a
//! [`Backend`] trait with a CPU implementation (the oracle) and a GPU implementation. The two
//! share every line except the step and the single-token decode.

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;

use crate::tokenizer::VOCAB;
use ferrum_model::sampling::{sample, Params, Rng};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_ref::optim::{Adam, Schedule};
pub use ferrum_ref::trainer::Engine;
use ferrum_ref::trainer::{StepConfig, StepStats};
use std::sync::{Arc, Mutex};
mod cpu;
mod gpu;
mod preset;
pub use preset::*;
mod progress;
pub use progress::*;
mod run_training;
pub mod validation;
pub use run_training::*;
mod generate_with;
pub use generate_with::*;
mod backend;
pub use backend::*;

#[cfg(test)]
mod tests;
