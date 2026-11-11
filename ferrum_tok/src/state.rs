// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/state.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Compact text serialization of the merge list stored inside model metadata.
use crate::error::bounded;
use crate::error::{InferError, Result};
use crate::ByteBpeTokenizer;

impl ByteBpeTokenizer {
    /// Serialise the merge list as `"a,b;c,d;…"`, optionally followed by a
    /// newline and the comma-separated special-token names (K3). An empty string
    /// is byte-level with no specials. Compact enough to embed in GGUF metadata.
    pub fn encode_state(&self) -> String {
        let merges = self
            .merges
            .iter()
            .map(|(a, b)| format!("{a},{b}"))
            .collect::<Vec<_>>()
            .join(";");
        if self.special.is_empty() {
            merges
        } else {
            format!("{merges}\n{}", self.special.join(","))
        }
    }

    /// Rebuild a tokenizer from [`encode_state`](Self::encode_state) output,
    /// including any special tokens.
    pub fn from_state(state: &str) -> Result<Self> {
        let mut tok = Self::byte_level();
        // Split the optional special-token section off the merge list.
        let (merge_part, special_part) = match state.split_once('\n') {
            Some((m, s)) => (m, Some(s)),
            None => (state, None),
        };
        // Each merge concatenates two earlier tokens, so a chain of merges can
        // double a token's length per entry: 40 entries (~400 bytes of state)
        // would build a 1 TiB vocab. Real vocabularies hold well under a byte
        // of token text per state byte, so cap the total.
        // ponytail: 64 B per state byte + 64 MiB; a tokenizer whose tokens
        // really exceed that (trained on one giant word) would need a lazier
        // vocab representation.
        let max_bytes = state.len().saturating_mul(64).saturating_add(64 << 20);
        let mut vocab_bytes = 0usize;
        if !merge_part.is_empty() {
            for entry in merge_part.split(';') {
                let (a, b) = entry
                    .split_once(',')
                    .ok_or_else(|| InferError::Format(format!("bad merge entry {entry:?}")))?;
                let a: u32 = a
                    .trim()
                    .parse()
                    .map_err(|_| InferError::ParseError(format!("bad merge id {a:?}")))?;
                let b: u32 = b
                    .trim()
                    .parse()
                    .map_err(|_| InferError::ParseError(format!("bad merge id {b:?}")))?;
                if a as usize >= tok.vocab.len() || b as usize >= tok.vocab.len() {
                    return Err(InferError::Format(format!(
                        "merge ({a},{b}) references undefined token (vocab {})",
                        tok.vocab.len()
                    )));
                }
                let len = tok.vocab[a as usize].len() + tok.vocab[b as usize].len();
                vocab_bytes = bounded(
                    vocab_bytes.saturating_add(len),
                    max_bytes,
                    "tokenizer state vocab bytes",
                )?;
                let merged =
                    [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
                tok.vocab.push(merged);
                tok.merges.push((a, b));
            }
        }
        if let Some(specials) = special_part {
            for name in specials.split(',').filter(|n| !n.is_empty()) {
                tok.add_special_token(name)?;
            }
        }
        tok.rebuild_ranks();
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::CORPUS;
    use crate::{ByteBpeTokenizer, TOK_EOS};

    #[test]
    fn state_roundtrip_preserves_encoding() {
        let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
        assert_eq!(restored.vocab_size(), tok.vocab_size());
        assert_eq!(restored.encode(CORPUS), tok.encode(CORPUS));
        assert_eq!(restored.decode(&restored.encode("lowest")), "lowest");
    }

    #[test]
    fn bad_state_errors() {
        assert!(ByteBpeTokenizer::from_state("not-a-pair").is_err());
        assert!(ByteBpeTokenizer::from_state("1,2;9999,3").is_err()); // undefined id
        assert!(ByteBpeTokenizer::from_state("x,y").is_err());
    }

    #[test]
    fn doubling_merge_chain_in_state_is_capped() {
        // "97,97;256,256;257,257;…": token 256+i is 2^(i+1) bytes, so 40
        // entries (~400 bytes) would build a 1 TiB vocab. Rejected near the
        // 64 MiB floor of the cap instead, fast.
        let chain = |n: u32| {
            std::iter::once("97,97".to_string())
                .chain((0..n - 1).map(|i| format!("{0},{0}", 256 + i)))
                .collect::<Vec<_>>()
                .join(";")
        };
        let t = std::time::Instant::now();
        let e = ByteBpeTokenizer::from_state(&chain(40)).err().unwrap();
        assert!(e.to_string().contains("tokenizer state vocab bytes"), "{e}");
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
        // A 20-long chain (1 MiB token, 2 MiB total) is within the cap.
        let tok = ByteBpeTokenizer::from_state(&chain(20)).unwrap();
        assert_eq!(tok.vocab.last().unwrap().len(), 1 << 20);
    }

    #[test]
    fn encode_matches_after_state_roundtrip_with_ranks() {
        // The rank map must be rebuilt by from_state so encoding is identical.
        let tok = ByteBpeTokenizer::train(CORPUS, 320).unwrap();
        let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
        for text in [CORPUS, "lowest newer news", "🌸 unseen"] {
            assert_eq!(
                restored.encode(text),
                tok.encode(text),
                "mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn special_tokens_survive_state_roundtrip() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        tok.add_standard_special_tokens();
        let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
        assert_eq!(restored.vocab_size(), tok.vocab_size());
        assert_eq!(restored.eos_id(), tok.eos_id());
        assert_eq!(restored.num_special(), 4);
        assert_eq!(restored.encode(CORPUS), tok.encode(CORPUS));

        // Specials also round-trip on a byte-level (no-merge) tokenizer.
        let mut bl = ByteBpeTokenizer::byte_level();
        bl.add_special_token(TOK_EOS).unwrap();
        let r = ByteBpeTokenizer::from_state(&bl.encode_state()).unwrap();
        assert_eq!(r.eos_id(), bl.eos_id());
        assert_eq!(r.vocab_size(), 256 + 1);
    }
}
