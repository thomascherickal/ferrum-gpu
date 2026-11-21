//! Tokenizer files and model metadata: the GGUF side of [`ferrum_tok::ByteBpeTokenizer`].
//!
//! Two layouts, both the ones ferrum-cpu writes (so a tokenizer file or a model from either side
//! loads on the other):
//! - a standalone **tokenizer file**: `general.architecture = "ferrum-tokenizer"`, the three
//!   `tokenizer.ggml.{tokens,merges,token_type}` arrays (a valid llama.cpp vocabulary) and
//!   `tokenizer.ferrum.state`, the lossless source of truth;
//! - the same keys inside a **model file**, read by [`from_gguf`] when the model carries them.
use ferrum_gguf::{Gguf, GgufBuilder, Value};
use ferrum_tok::ByteBpeTokenizer;
mod tokenizer_arch;
pub use tokenizer_arch::*;
mod from_gguf;
pub use from_gguf::*;

#[cfg(test)]
mod tests;
