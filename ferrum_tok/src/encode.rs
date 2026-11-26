// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/encode.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Encoding text to token ids (rank-based merging) and decoding back.
use crate::pretokenize::pretokenize;
use crate::train::merge_pair;
use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP};
mod chunks;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_unknown_ids;
