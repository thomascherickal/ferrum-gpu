// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/encode.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Encoding text to token ids (rank-based merging) and decoding back.
use crate::pretokenize::pretokenize;
use crate::train::merge_pair;
use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP};

impl ByteBpeTokenizer {
    /// Rank-based BPE encode of one pre-tokenized chunk (K1): repeatedly apply
    /// the lowest-rank (earliest-learned) applicable merge until none applies —
    /// `O(applied merges × len)` instead of rescanning the whole text once per
    /// learned merge.
    fn encode_chunk(&self, bytes: &[u8], out: &mut Vec<usize>) {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        while ids.len() >= 2 {
            // Lowest-rank adjacent pair present in the sequence.
            let mut best: Option<(usize, (u32, u32))> = None;
            for w in ids.windows(2) {
                let pair = (w[0], w[1]);
                if let Some(&rank) = self.ranks.get(&pair) {
                    if best.is_none_or(|(br, _)| rank < br) {
                        best = Some((rank, pair));
                    }
                }
            }
            let Some((rank, pair)) = best else { break };
            ids = merge_pair(&ids, pair, (BASE_VOCAB + rank) as u32);
        }
        out.extend(ids.into_iter().map(|i| i as usize));
    }

    /// Like [`encode`](Self::encode), but every [`DOC_SEP`] becomes the EOS id
    /// when this tokenizer has one. Without EOS it is exactly `encode`.
    pub fn encode_docs(&self, text: &str) -> Vec<usize> {
        let Some(eos) = self.eos_id() else {
            return self.encode(text);
        };
        let mut ids = Vec::new();
        for (i, doc) in text.split(DOC_SEP).enumerate() {
            if i > 0 {
                ids.push(eos);
            }
            ids.extend(self.encode(doc));
        }
        ids
    }

    /// Encode text to token IDs: pre-tokenize at whitespace boundaries (K2),
    /// then rank-based BPE-encode each chunk (K1) and concatenate.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            self.encode_chunk(chunk.as_bytes(), &mut out);
        }
        out
    }

    /// Encode `text`, optionally framed with the registered BOS/EOS tokens
    /// (I3/K3). Framing is skipped silently for any token not registered.
    pub fn encode_framed(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if bos {
            if let Some(b) = self.bos_id() {
                ids.push(b);
            }
        }
        ids.extend(self.encode(text));
        if eos {
            if let Some(e) = self.eos_id() {
                ids.push(e);
            }
        }
        ids
    }

    /// Decode token IDs back to text. Unknown IDs are skipped; byte sequences
    /// that are not valid UTF-8 decode lossily (U+FFFD). Special tokens map to
    /// empty byte strings, so they vanish from decoded text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.vocab.get(id) {
                bytes.extend_from_slice(tok);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::CORPUS;
    use crate::ByteBpeTokenizer;

    #[test]
    fn trained_roundtrip_exact() {
        let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        for text in [CORPUS, "lowest news", "unseen zzz 🌸 text"] {
            assert_eq!(
                tok.decode(&tok.encode(text)),
                text,
                "roundtrip failed for {text:?}"
            );
        }
    }

    #[test]
    fn rank_based_encode_is_greedy_lowest_rank_first() {
        // Train so "lo","ow" (hence "low") are learned; encoding "low" must use
        // the merged token, i.e. fewer tokens than its 3 bytes.
        let tok = ByteBpeTokenizer::train("low low low low low", 300).unwrap();
        let ids = tok.encode("low");
        assert!(ids.len() < 3, "expected a merged 'low' token, got {ids:?}");
        assert_eq!(tok.decode(&ids), "low");
    }

    #[test]
    fn encode_uses_learned_merges_across_words_independently() {
        let tok = ByteBpeTokenizer::train("hello hello world world hello world", 320).unwrap();
        // Two words → each encodes via its own within-word merges; concatenation
        // round-trips and the space is its own (unmerged) token between them.
        let ids = tok.encode("hello world");
        assert_eq!(tok.decode(&ids), "hello world");
        assert!(ids.len() <= "hello world".len());
    }

    #[test]
    fn encode_framed_adds_bos_and_eos() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        tok.add_standard_special_tokens();
        let bare = tok.encode("lower");
        let framed = tok.encode_framed("lower", true, true);
        assert_eq!(framed.first(), Some(&tok.bos_id().unwrap()));
        assert_eq!(framed.last(), Some(&tok.eos_id().unwrap()));
        assert_eq!(&framed[1..framed.len() - 1], bare.as_slice());
        // Each side is optional.
        assert_eq!(tok.encode_framed("lower", false, false), bare);
        assert_eq!(
            tok.encode_framed("lower", true, false)[0],
            tok.bos_id().unwrap()
        );
        assert_eq!(
            tok.encode_framed("lower", true, false).len(),
            bare.len() + 1
        );
        assert_eq!(
            tok.encode_framed("lower", false, true).len(),
            bare.len() + 1
        );
        // Without registered specials, framing is a no-op.
        let plain = ByteBpeTokenizer::byte_level();
        assert_eq!(
            plain.encode_framed("lower", true, true),
            plain.encode("lower")
        );
    }

    #[test]
    fn encode_docs_separator_positions_and_missing_eos() {
        let tok = ByteBpeTokenizer::train(&"ab ab ab ".repeat(10), 270).unwrap();
        let eos = tok.eos_id().unwrap();
        // Leading, trailing, adjacent and absent separators.
        assert_eq!(tok.encode_docs("<|endoftext|>"), vec![eos]);
        assert_eq!(
            tok.encode_docs("<|endoftext|><|endoftext|>"),
            vec![eos, eos]
        );
        let mixed = tok.encode_docs("<|endoftext|>ab<|endoftext|>");
        assert_eq!(mixed.first(), Some(&eos));
        assert_eq!(mixed.last(), Some(&eos));
        assert_eq!(tok.encode_docs("ab ab"), tok.encode("ab ab"));
        assert!(tok.encode_docs("").is_empty());
        // A tokenizer without EOS keeps the separator as literal text.
        let bare = ByteBpeTokenizer::byte_level();
        assert!(bare.eos_id().is_none());
        assert_eq!(
            bare.encode_docs("x<|endoftext|>y"),
            bare.encode("x<|endoftext|>y")
        );
        assert_eq!(
            bare.decode(&bare.encode_docs("x<|endoftext|>y")),
            "x<|endoftext|>y"
        );
    }
}
