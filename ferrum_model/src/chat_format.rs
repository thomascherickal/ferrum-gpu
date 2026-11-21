//! Chat-prompt formatting for instruct GGUFs.
//!
//! Provenance: ported from ferrum-cpu `ferrum_core/src/chat_format.rs` (commit 56a9d3a); tests
//! changed to build tokenizers with our GGUF writer.
//!
//! No Jinja: the `tokenizer.chat_template` string is *fingerprinted* against
//! the four formats small llama/qwen models actually use, with a vocab-based
//! fallback when the key is absent. `render` returns text/special segments so
//! the runner can map each special token to its exact vocab id instead of
//! letting byte-BPE shred it. Single-turn only, by design.
use crate::gguf_tok::GgufTokenizer;
mod chatformat;
pub use chatformat::*;
mod render;
pub use render::*;

#[cfg(test)]
mod tests;
