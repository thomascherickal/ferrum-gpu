// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/special.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Reserved special tokens (BOS / EOS / PAD / UNK and custom).
use crate::error::{InferError, Result};
use crate::{ByteBpeTokenizer, BASE_VOCAB, TOK_BOS, TOK_EOS, TOK_PAD, TOK_UNK};

impl ByteBpeTokenizer {
    /// Append a named special token, returning its reserved id (above the byte +
    /// merge range). Idempotent: re-adding an existing name returns its id.
    /// Names may not contain `,`, `;`, or newlines (the state delimiters).
    pub fn add_special_token(&mut self, name: &str) -> Result<usize> {
        if name.is_empty() || name.contains(',') || name.contains(';') || name.contains('\n') {
            return Err(InferError::Format(
                "special-token name must be non-empty and free of ',', ';', and newlines".into(),
            ));
        }
        if let Some(id) = self.special_id(name) {
            return Ok(id);
        }
        let id = self.vocab.len();
        self.vocab.push(Vec::new()); // decodes to nothing
        self.special.push(name.to_string());
        Ok(id)
    }

    /// Register the four conventional special tokens ([`TOK_BOS`], [`TOK_EOS`],
    /// [`TOK_PAD`], [`TOK_UNK`]) and return their ids in that order.
    pub fn add_standard_special_tokens(&mut self) -> [usize; 4] {
        [TOK_BOS, TOK_EOS, TOK_PAD, TOK_UNK].map(|n| self.add_special_token(n).unwrap())
    }

    /// The id of a registered special token, if present.
    pub fn special_id(&self, name: &str) -> Option<usize> {
        self.special
            .iter()
            .position(|n| n == name)
            .map(|i| BASE_VOCAB + self.merges.len() + i)
    }

    /// Convenience accessors for the conventional special tokens.
    pub fn bos_id(&self) -> Option<usize> {
        self.special_id(TOK_BOS)
    }

    /// Id of the end-of-sequence special token, if registered.
    pub fn eos_id(&self) -> Option<usize> {
        self.special_id(TOK_EOS)
    }

    /// Id of the padding special token, if registered.
    pub fn pad_id(&self) -> Option<usize> {
        self.special_id(TOK_PAD)
    }

    /// Id of the unknown special token, if registered.
    pub fn unk_id(&self) -> Option<usize> {
        self.special_id(TOK_UNK)
    }

    /// Number of registered special tokens.
    pub fn num_special(&self) -> usize {
        self.special.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::{small_tok, CORPUS};
    use crate::{ByteBpeTokenizer, TOK_EOS};

    #[test]
    fn special_tokens_get_reserved_ids_above_vocab() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        let base = tok.vocab_size();
        // train() already reserved EOS as the last id; the rest are appended.
        let [bos, eos, pad, unk] = tok.add_standard_special_tokens();
        assert_eq!([bos, eos, pad, unk], [base, base - 1, base + 1, base + 2]);
        assert_eq!(tok.vocab_size(), base + 3);
        assert_eq!(tok.eos_id(), Some(eos));
        assert_eq!(tok.bos_id(), Some(bos));
        assert_eq!(tok.num_special(), 4);
        // Adding an existing special is idempotent.
        assert_eq!(tok.add_special_token(TOK_EOS).unwrap(), eos);
        assert_eq!(tok.vocab_size(), base + 3);
    }

    #[test]
    fn special_ids_are_disjoint_from_text_and_decode_to_empty() {
        let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        tok.add_standard_special_tokens();
        // Ordinary text never encodes to a special id.
        let ids = tok.encode("low lowest newer");
        let max_text_id = tok.vocab_size() - tok.num_special();
        assert!(
            ids.iter().all(|&i| i < max_text_id),
            "text encoded to a special id"
        );
        // Special tokens decode to nothing, so they vanish from text.
        let eos = tok.eos_id().unwrap();
        assert_eq!(tok.decode(&[eos]), "");
        let mixed = tok.encode_framed("low", true, true);
        assert_eq!(
            tok.decode(&mixed),
            "low",
            "framing must not alter decoded text"
        );
    }

    #[test]
    fn add_special_token_rejects_delimiter_names() {
        let mut tok = ByteBpeTokenizer::byte_level();
        assert!(tok.add_special_token("").is_err());
        assert!(tok.add_special_token("a,b").is_err());
        assert!(tok.add_special_token("a;b").is_err());
        assert!(tok.add_special_token("a\nb").is_err());
        assert!(tok.add_special_token(TOK_EOS).is_ok());
    }

    #[test]
    fn standard_special_tokens_ids_and_gguf_types() {
        let mut t = small_tok();
        let n_text = t.vocab_size() - 1;
        let ids = t.add_standard_special_tokens();
        assert_eq!(ids, [n_text + 1, n_text, n_text + 2, n_text + 3]);
        assert_eq!(t.pad_id(), Some(n_text + 2));
        assert_eq!(t.unk_id(), Some(n_text + 3));
        let types = t.gguf_token_types();
        assert_eq!(types.len(), t.vocab_size());
        assert!(types[..n_text].iter().all(|&x| x == 1));
        assert!(types[n_text..].iter().all(|&x| x == 3));
        let toks = t.gguf_tokens();
        assert_eq!(
            &toks[n_text..],
            ["<|endoftext|>", "<bos>", "<pad>", "<unk>"]
        );
        let back = ByteBpeTokenizer::from_gguf_vocab(&toks, &t.gguf_merges(), &types).unwrap();
        assert_eq!(back.encode_state(), t.encode_state());
        assert_eq!(back.unk_id(), t.unk_id());
    }
}
