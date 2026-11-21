//! `generate`, `run-gguf`, `eval` and `info` for GPT-2 files.
use super::score::mean_loss;
use crate::commands::device::Workload;
use crate::commands::flags::Args;
use crate::commands::{app, device_line, mem_mb, say};
use crate::session::Session;
use ferrum_gguf::Gguf;
use ferrum_model::gpt2_gguf::from_gguf;
use ferrum_model::sampling::{sample, Params, Rng};
use ferrum_model::tokenizer_gguf;
use ferrum_ref::gpt2::Gpt2;
use ferrum_tok::ByteBpeTokenizer;
use std::io::Write;
mod loaded;
pub use loaded::*;
mod run_m;
pub use run_m::*;
mod eval_m;
pub use eval_m::*;
