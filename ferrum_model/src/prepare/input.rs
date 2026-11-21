//! Reading the inputs of a corpus run: which files, and their documents streamed in fixed-size
//! chunks.
//!
//! Provenance: ported from ferrum-cpu `slm_cli/src/prepare/input.rs` (commit 56a9d3a).
use crate::sha256::Sha256;
use std::io::Read;
use std::path::{Path, PathBuf};
mod expand_inputs;
pub use expand_inputs::*;
mod for_each_raw_doc;
pub use for_each_raw_doc::*;

#[cfg(test)]
mod tests;
