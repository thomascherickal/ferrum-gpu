//! `generate`, `run-gguf` and `eval` for imported Llama/Qwen files that carry their own tokenizer
//! (`tokenizer.ggml.*`). Files without one are the byte-level models of `cmds::generate`.
//!
//! An instruct model is prompted through its chat template (ChatML, Llama-3, `[INST]`, Zephyr),
//! unless `--raw` or `--ids` is given; the reply ends at the end-of-turn token. In chat mode only
//! the reply is printed; in raw mode the prompt is printed first, as for GPT-2 files.
use super::gpt2::gen::{opts, print_run};
use super::{app, device_line, gate, mean_loss_ids, mem_mb, say, shape};
use crate::commands::device::Workload;
use crate::commands::flags::Args;
use crate::commands::preflight;
use ferrum_gguf::Gguf;
use ferrum_model::chat_format::{detect, encode_segments, render, stop_token, ChatFormat};
use ferrum_model::gguf_tok::GgufTokenizer;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::Llama;
use std::io::Write;
mod loaded;
pub use loaded::*;
mod generate;
pub use generate::*;
mod eval;
pub use eval::*;

#[cfg(test)]
mod tests;
