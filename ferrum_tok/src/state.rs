// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/state.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Compact text serialization of the merge list stored inside model metadata.
use crate::error::bounded;
use crate::error::{InferError, Result};
use crate::ByteBpeTokenizer;
mod part;

#[cfg(test)]
mod tests;
