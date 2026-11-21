use super::*;

impl ByteBpeTokenizer {
    /// Rebuild from the three GGUF arrays (inverse of the exporters). Expects
    /// Ferrum's layout: 256 byte tokens, then one token per merge in rank
    /// order, then control tokens. Errors on any deviation, except that the
    /// `token_type` of the text tokens (the first `256 + merges`) is not
    /// inspected: foreign GPT-2 GGUFs label them inconsistently (normal, byte,
    /// unused) and the `tokens`/`merges` arrays alone define them. Only the
    /// tokens after the merges must be control (3).
    pub fn from_gguf_vocab(
        tokens: &[String],
        merges: &[String],
        token_types: &[i32],
    ) -> Result<Self> {
        let bad = |m: String| InferError::Format(format!("gguf vocab: {m}"));
        let n_text = BASE_VOCAB + merges.len();
        if tokens.len() < n_text || token_types.len() != tokens.len() {
            return Err(bad(format!(
                "{} tokens / {} types for {} merges",
                tokens.len(),
                token_types.len(),
                merges.len()
            )));
        }
        let ids: HashMap<&str, u32> = tokens[..n_text]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i as u32))
            .collect();
        let mut tok = Self::byte_level();
        for (b, t) in tokens[..BASE_VOCAB].iter().enumerate() {
            if *t != byte_to_unicode(b as u8).to_string() {
                return Err(bad(format!("token {b} is not byte {b}")));
            }
        }
        for (i, m) in merges.iter().enumerate() {
            // llama.cpp: split at the first space searching from index 1.
            let from = m.chars().next().map_or(0, char::len_utf8);
            let (l, r) = m[from..]
                .find(' ')
                .map(|j| (&m[..from + j], &m[from + j + 1..]))
                .ok_or_else(|| bad(format!("bad merge {m:?}")))?;
            let (Some(&a), Some(&b)) = (ids.get(l), ids.get(r)) else {
                return Err(bad(format!("merge {m:?} has a part that is not a token")));
            };
            if a as usize >= BASE_VOCAB + i || b as usize >= BASE_VOCAB + i {
                return Err(bad(format!("merge {m:?} uses a later token")));
            }
            let merged = [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
            let want: String = merged.iter().map(|&x| byte_to_unicode(x)).collect();
            if tokens[BASE_VOCAB + i] != want {
                return Err(bad(format!("token {} is not merge {m:?}", BASE_VOCAB + i)));
            }
            tok.vocab.push(merged);
            tok.merges.push((a, b));
        }
        for (k, (name, ty)) in tokens[n_text..]
            .iter()
            .zip(&token_types[n_text..])
            .enumerate()
        {
            if *ty != 3 {
                return Err(bad(format!("token {name:?} after merges is not control")));
            }
            if tok.add_special_token(name)? != n_text + k {
                return Err(bad(format!("duplicate special token {name:?}")));
            }
        }
        tok.rebuild_ranks();
        Ok(tok)
    }
}
