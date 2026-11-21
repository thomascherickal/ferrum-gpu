// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/train.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! BPE merge learning.
use crate::error::{InferError, Result};
use crate::pretokenize::pretokenize;
use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP, TOK_EOS};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
mod corpus;
mod from_reader;
mod merge;
mod merge_step;
use merge_step::{apply_deltas, merge_in_words};
mod word_counts;
pub(crate) use merge::merge_pair;

#[cfg(test)]
mod tests;
