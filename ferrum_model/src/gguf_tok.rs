//! The tokenizer stored in an imported Llama/Qwen GGUF (`tokenizer.ggml.*`), so those models can be
//! driven with text. Two families: byte-level BPE (`gpt2`/`bpe`: Llama-3, Qwen2) and SentencePiece
//! (`llama`/`spm`).
//!
//! Provenance: ported from ferrum-cpu `ferrum_core/src/gguf_tokenizer.rs` (commit 56a9d3a) with the
//! metadata access rewritten for our `ferrum_gguf`. Limits, unchanged from the original: the BPE
//! pre-tokenizer is a dependency-free approximation of the GPT-2/Qwen regex (it can differ on
//! exotic whitespace), and SPM encoding is a greedy longest match, not SentencePiece's Viterbi
//! search. `decode` is exact for both. Pass token ids (`--ids`) where exactness matters.
use ferrum_gguf::{Gguf, Value};
use std::collections::HashMap;
mod spm_space;
pub use spm_space::*;
mod bpe;
mod loading;
mod pretok;
mod queries;
mod spm;
use pretok::*;

#[cfg(test)]
mod tests;
