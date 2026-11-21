//! Reading the CLI's `--verbose` progress lines into events.
//!
//! The formats are a contract with `ferrum_gpu_cli` (`cmds/gpt2/hooks.rs`):
//! `step 20/200 loss 3.1234 grad_norm 0.512 tokens/s 4800 elapsed 4.2s eta 38s` and
//! `epoch 3 val loss 2.9876 (best)`, and the result line of `eval`:
//! `tokens: 4096 loss: 2.3456 perplexity: 10.442`.
use crate::json::Json;
mod event;
pub use event::*;
mod part;
