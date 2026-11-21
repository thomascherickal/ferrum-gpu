// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/pretokenize.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! GPT-2 style pre-tokenization and the byte-to-unicode table.

mod byte_to_unicode;
pub use byte_to_unicode::*;
mod word_split;
pub(crate) use word_split::*;

#[cfg(test)]
mod tests;
