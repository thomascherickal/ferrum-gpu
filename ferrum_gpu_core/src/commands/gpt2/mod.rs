//! The GPT-2 path of the CLI: what ferrum-cpu's `train_transformer` does. A GPT-2 model with
//! learned positions, LayerNorm and GELU, a byte-level BPE tokenizer stored inside the GGUF, so
//! llama.cpp and ferrum-cpu can run the file.
//!
//! Sizes, defaults and flag spellings follow ferrum-cpu (docs/parity.md); the `llama` path is the
//! one used for imported Llama/Qwen files and `--arch llama`.

pub use gen::{eval, generate, is_gpt2, load, shape_of};
pub use tokenize::tokenize;
pub use train::train;

use crate::commands::flags::{many, switch, val, Args, Flag};
use ferrum_model::gpt2_gguf::Quant;
#[cfg(test)]
pub(crate) mod error_paths;
mod extras;
pub mod gen;
mod hooks;
pub mod prepare;
mod preset;
pub use preset::*;
mod quant_names;
pub use quant_names::*;
mod train_flags;
pub use train_flags::*;
mod only_gpt2;
mod score;
mod tokenize;
mod train;
pub use only_gpt2::*;

#[cfg(test)]
mod tests;
