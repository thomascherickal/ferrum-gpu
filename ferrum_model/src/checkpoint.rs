// Provenance: the "one GGUF file holding p{i}.data / p{i}.m / p{i}.v plus step and run keys"
// idea follows ferrum-cpu's training checkpoints (`train_transformer/checkpoint.rs`, commit
// 56a9d3a). The kind below is our own and is NOT interchangeable with ferrum-cpu's checkpoints.
//! Training checkpoints as GGUF files: weights, Adam moments, step and loop position.
use ferrum_gguf::reader::narrow;
use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value};
use ferrum_ref::gpt2::{param_lens, Config, Gpt2};
use ferrum_ref::trainer::{Checkpoint, RunShape, Snapshot};
use std::io::Write;
use std::path::Path;
mod kind;
pub use kind::*;
mod from_bytes;
pub use from_bytes::*;
mod atomic_write;
pub use atomic_write::*;

#[cfg(test)]
mod tests;
