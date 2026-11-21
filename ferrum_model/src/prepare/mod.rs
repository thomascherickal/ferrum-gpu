//! `prepare`: stream raw text, clean it per document, drop duplicates and out-of-range lengths,
//! split train/val/test by a seeded content hash, and write a manifest that pins every input and
//! output by sha256 — a deterministic, DVC-friendly data stage.
//!
//! Provenance: ported from ferrum-cpu `slm_cli/src/prepare/mod.rs` (commit 56a9d3a). Same cleaning
//! order, same split hash and the same `manifest.json` layout, so equal inputs and options give
//! equal `train.txt`/`val.txt`/`test.txt` bytes. Not ported: the progress callback and the stop
//! hook (the CLI has no use for them).

pub use input::expand_inputs;

use crate::sha256::Sha256;
use json::{obj, Json};
use split::{part, Split};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
mod clean;
mod input;
mod json;
mod prepareoptions;
mod split;
pub use prepareoptions::*;
mod build_m;
use build_m::*;

#[cfg(test)]
mod tests;
