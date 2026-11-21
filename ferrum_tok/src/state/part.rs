use super::*;

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
