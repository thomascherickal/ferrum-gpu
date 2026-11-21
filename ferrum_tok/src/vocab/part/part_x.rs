use super::*;

impl ByteBpeTokenizer {
    /// `tokenizer.ggml.tokens`: each token's bytes mapped through
    /// [`byte_to_unicode`]; special tokens by name.
    pub fn gguf_tokens(&self) -> Vec<String> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| match i.checked_sub(n_text) {
                Some(s) => self.special[s].clone(),
                None => self.vocab[i].iter().map(|&b| byte_to_unicode(b)).collect(),
            })
            .collect()
    }

    /// `tokenizer.ggml.merges` in rank order: `"<left> <right>"`.
    pub fn gguf_merges(&self) -> Vec<String> {
        let toks = self.gguf_tokens();
        self.merges
            .iter()
            .map(|&(a, b)| format!("{} {}", toks[a as usize], toks[b as usize]))
            .collect()
    }

    /// `tokenizer.ggml.token_type`: 1 = normal, 3 = control (specials).
    pub fn gguf_token_types(&self) -> Vec<i32> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| if i < n_text { 1 } else { 3 })
            .collect()
    }
}
