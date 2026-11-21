//! `train` for the GPT-2 path: tokenizer, model, loop, checkpoints, GGUF out.
use super::extras;
use super::gen::Loaded;
use super::hooks::{Run, Validation};
use super::{quant, Sizes};
use crate::commands::flags::Args;
use crate::commands::{device_line, gate, mem_mb, say};
use ferrum_model::checkpoint::{atomic_write, from_bytes};
use ferrum_model::gpt2_gguf::to_gguf_with;
use ferrum_model::sha256::sha256_hex;
use ferrum_model::token_file::{is_token_file, TokenFile};
use ferrum_model::tokenizer_gguf;
use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::optim::Schedule;
use ferrum_ref::trainer::{
    steps_per_epoch, train as run_loop, window_starts, Prefix, RunShape, TokenSource, TrainConfig,
};
use ferrum_tok::ByteBpeTokenizer;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
mod tokenizer_m;
mod train_m;
use tokenizer_m::*;
pub use train_m::*;
