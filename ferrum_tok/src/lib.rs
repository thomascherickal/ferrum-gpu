// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/mod.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! Byte-level Byte-Pair Encoding (BPE) tokenizer, written in `std` (the
//! pre-tokenizer uses `unicode-general-category`).
//!
//! The base vocabulary is the 256 single bytes, so *any* text — emoji,
//! Cyrillic, CJK, control characters — round-trips without an unknown-token
//! escape hatch. Training greedily merges the most frequent adjacent token
//! pair until the requested vocabulary size is reached (or no pair repeats).
//!
//! ```rust
//! use ferrum_tok::ByteBpeTokenizer;
//!
//! let corpus = "low lower lowest low low";
//! let tok = ByteBpeTokenizer::train(corpus, 300).unwrap();
//! let ids = tok.encode("lowest");
//! assert_eq!(tok.decode(&ids), "lowest");
//! ```

/// Verbose progress output is not part of this crate (ferrum-cpu prints from here); the calls
/// stay so the copied trainer is unchanged, and they compile to nothing.
// ponytail: no-op; a CLI `--verbose` for tokenizer training would route this to stderr.
macro_rules! vprintln {
    ($($t:tt)*) => {};
}

mod encode;
pub mod error;
mod pretokenize;
mod special;
mod state;
#[cfg(test)]
mod test_support;
mod train;
mod vocab;

pub use self::pretokenize::byte_to_unicode;

use std::collections::HashMap;

/// Number of base tokens (one per byte value).
const BASE_VOCAB: usize = 256;

/// A trained byte-level BPE tokenizer.
///
/// The full state is the ordered merge list: token `256 + i` is defined as
/// the concatenation of merge `i`'s pair, so [`encode_state`](Self::encode_state)
/// serialises merges only and [`from_state`](Self::from_state) rebuilds the
/// vocabulary deterministically.
#[derive(Debug, Clone)]
pub struct ByteBpeTokenizer {
    /// id → byte string. ids 0..256 are the single bytes; special tokens (if
    /// any) occupy the highest ids and map to empty byte strings.
    vocab: Vec<Vec<u8>>,
    /// Learned merges in order; merge `i` creates token id `256 + i`.
    merges: Vec<(u32, u32)>,
    /// `pair → merge rank` (= index in `merges`), for the O(applied) rank-based
    /// encoder (K1). Rebuilt whenever `merges` changes.
    ranks: HashMap<(u32, u32), usize>,
    /// Named special tokens (BOS/EOS/PAD/UNK, …) in id order (K3). Their ids are
    /// reserved **above** the byte + merge range, so adding them never collides
    /// with bytes or merges. They never arise from encoding ordinary text and
    /// decode to the empty string.
    special: Vec<String>,
}

/// Conventional names for the four standard special tokens (K3).
/// Name of the beginning-of-sequence special token.
pub const TOK_BOS: &str = "<bos>";

/// Name of the end-of-sequence special token.
pub const TOK_EOS: &str = "<|endoftext|>";

/// Name of the padding special token.
pub const TOK_PAD: &str = "<pad>";

/// Name of the unknown special token.
pub const TOK_UNK: &str = "<unk>";

/// Document separator used by prepared corpora (`train_transformer prepare`)
/// and TinyStories. [`ByteBpeTokenizer::encode_docs`] maps it to the EOS special
/// token, and [`ByteBpeTokenizer::train_with_docs`] keeps it out of the merges.
pub const DOC_SEP: &str = TOK_EOS; // "<|endoftext|>" since Phase 0-G GT

impl ByteBpeTokenizer {
    /// A tokenizer with no merges: pure byte-level (vocab = 256).
    pub fn byte_level() -> Self {
        Self {
            vocab: (0..=255u8).map(|b| vec![b]).collect(),
            merges: Vec::new(),
            ranks: HashMap::new(),
            special: Vec::new(),
        }
    }

    /// Rebuild the `pair → rank` map from the current merge list.
    fn rebuild_ranks(&mut self) {
        self.ranks = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
    }

    /// Total number of token ids: base bytes, merges and special tokens.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// The raw bytes of one token, if the ID is in range.
    pub fn token_bytes(&self, id: usize) -> Option<&[u8]> {
        self.vocab.get(id).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::ByteBpeTokenizer;

    #[test]
    fn byte_level_roundtrip() {
        let tok = ByteBpeTokenizer::byte_level();
        let text = "hello 🌸 мир\n";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len()); // one token per byte
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn token_bytes_lookup() {
        let tok = ByteBpeTokenizer::byte_level();
        assert_eq!(tok.token_bytes(b'a' as usize), Some(&b"a"[..]));
        assert_eq!(tok.token_bytes(999), None);
    }
}
