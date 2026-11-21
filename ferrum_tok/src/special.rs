// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/special.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Reserved special tokens (BOS / EOS / PAD / UNK and custom).
use crate::error::{InferError, Result};
use crate::{ByteBpeTokenizer, BASE_VOCAB, TOK_BOS, TOK_EOS, TOK_PAD, TOK_UNK};
mod special_tokens;

#[cfg(test)]
mod tests;
