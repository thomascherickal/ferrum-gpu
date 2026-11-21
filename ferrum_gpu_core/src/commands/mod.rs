//! The commands. Each writes its report to `out` and returns an error string instead of exiting,
//! so tests drive them with the CPU backend and a `Vec<u8>`.
use crate::commands::device::{decide, Workload};
use crate::commands::flags::{switch, val, Args, Flag};
use crate::engine::{fresh_model, preset, GenOpts, Progress, TrainSpec};
use crate::session::{DeviceChoice, Session};
use crate::tokenizer::{decode, encode};
use ferrum_gguf::Gguf;
use ferrum_model::checkpoint::atomic_write;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::Llama;
use ferrum_wgpu::Gpu;
use std::io::Write;
use std::sync::{Arc, Mutex};

pub mod combined;
pub mod device;
pub mod flags;
pub mod gpt2;
pub mod lab;
pub mod llm;
pub mod preflight;
pub mod runtimes;
pub mod sysmem;
mod usage;
pub use usage::*;
mod dispatch_m;
mod gate_m;
pub use gate_m::*;
mod train_m;
use dispatch_m::*;
pub use train_m::*;

#[cfg(test)]
mod tests;
