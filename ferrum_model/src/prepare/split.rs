//! One output split (`train.txt`, `val.txt`, `test.txt`): a `*.part` file hashed and counted as it
//! is written.
//!
//! Provenance: ported from ferrum-cpu `slm_cli/src/prepare/split.rs` (commit 56a9d3a).
use super::json::{obj, Json};
use crate::sha256::Sha256;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
mod join;
pub(crate) use join::*;

#[cfg(test)]
mod tests;
