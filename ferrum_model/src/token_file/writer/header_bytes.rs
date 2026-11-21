use super::*;

/// The GGUF header (metadata, directory, alignment padding) for `n` tokens in `docs` documents.
/// Every count is fixed-width, so the length does not depend on `n` or `docs`: the writer emits a
/// placeholder first and rewrites it in place when the counts are known.
pub fn header_bytes(meta: &TokenFileMeta, n: u64, docs: u64) -> Vec<u8> {
    let (ty, _) = element_type(meta.vocab_size);
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(TOKEN_FILE_ARCH.into()));
    b.meta("ferrum.tokens.version", Value::U32(VERSION));
    b.meta(
        "ferrum.tokens.vocab_size",
        Value::U32(meta.vocab_size as u32),
    );
    b.meta("ferrum.tokens.token_count", Value::U64(n));
    b.meta("ferrum.tokens.document_count", Value::U64(docs));
    b.meta(
        "ferrum.tokens.tokenizer_sha256",
        Value::Str(meta.tokenizer_sha256.clone()),
    );
    if let Some(eos) = meta.eos_id {
        b.meta("ferrum.tokens.eos_id", Value::U32(eos));
    }
    // a tensor that claims `n` ids but carries no data serialises to exactly the header
    b.tensor_raw(TENSOR, &[n], ty, Vec::new());
    b.into_bytes()
}

/// Receives the ids of a token file being written (see [`write`]).
pub struct TokenSink<'a> {
    pub(super) w: BufWriter<&'a mut std::fs::File>,
    pub(super) vocab: usize,
    pub(super) eos: Option<u32>,
    pub(super) wide: bool,
    pub(super) tokens: u64,
    pub(super) eos_seen: u64,
}

impl TokenSink<'_> {
    /// Ids pushed so far.
    pub fn len(&self) -> u64 {
        self.tokens
    }

    /// True when no id was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Append `ids`; one outside the vocabulary is an error.
    pub fn push(&mut self, ids: &[u32]) -> Result<(), String> {
        for &id in ids {
            if id as usize >= self.vocab {
                return Err(format!(
                    "token {} is id {id}, outside the vocabulary of {}",
                    self.tokens, self.vocab
                ));
            }
            self.eos_seen += u64::from(self.eos == Some(id));
            let r = if self.wide {
                self.w.write_all(&id.to_le_bytes())
            } else {
                self.w.write_all(&(id as i16).to_le_bytes())
            };
            r.map_err(|e| e.to_string())?;
            self.tokens += 1;
        }
        Ok(())
    }
}
