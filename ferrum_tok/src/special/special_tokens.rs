use super::*;

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
