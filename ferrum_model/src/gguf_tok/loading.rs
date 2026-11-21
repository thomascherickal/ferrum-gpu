use super::*;

impl GgufTokenizer {
    /// Reconstruct the tokenizer from a parsed GGUF's metadata. Errors if the
    /// file declares no tokenizer or an unsupported family.
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let model_str = g
            .string("tokenizer.ggml.model")
            .map_err(|_| "GGUF has no tokenizer.ggml.model metadata".to_string())?;
        let model = match model_str {
            "gpt2" | "bpe" => TokModel::Bpe,
            "llama" | "spm" => TokModel::Spm,
            other => {
                return Err(format!(
                    "unsupported GGUF tokenizer model '{other}' (have: gpt2/bpe, llama/spm)"
                ))
            }
        };

        let tokens =
            string_array(g, "tokenizer.ggml.tokens").ok_or("GGUF missing tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err("GGUF tokenizer.ggml.tokens is empty".into());
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = HashMap::new();
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.entry(tok.clone()).or_insert(id as u32);
            if let Some(b) = parse_byte_token(tok) {
                byte_to_id.insert(b, id as u32);
            }
        }

        // BPE merges: "left right" piece pairs, in priority order. Stored under a
        // `left\0right` joined key (see the `merges` field docs).
        let mut merges = HashMap::new();
        if let Some(pairs) = string_array(g, "tokenizer.ggml.merges") {
            for (rank, m) in pairs.iter().enumerate() {
                if let Some(sp) = m.find(' ') {
                    let key = format!("{}\u{0}{}", &m[..sp], &m[sp + 1..]);
                    merges.entry(key).or_insert(rank as u32);
                }
            }
        }

        // Longest token (in chars) for the SPM longest-match bound.
        let max_token_len = tokens
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);

        let bos = g.uint("tokenizer.ggml.bos_token_id").ok().map(|u| u as u32);
        let eos = g.uint("tokenizer.ggml.eos_token_id").ok().map(|u| u as u32);

        let add_bos = match g.get("tokenizer.ggml.add_bos_token") {
            Some(Value::Bool(b)) => *b,
            // llama.cpp's per-pre-tokenizer defaults (llama-vocab.cpp, b11521)
            _ => {
                model == TokModel::Spm
                    || matches!(
                        g.string("tokenizer.ggml.pre").ok(),
                        Some(
                            "llama3"
                                | "llama-v3"
                                | "llama-bpe"
                                | "falcon3"
                                | "falcon-h1"
                                | "pixtral"
                                | "midm-2.0"
                                | "lfm2"
                                | "jina-v5-nano"
                                | "tekken"
                                | "chameleon"
                        )
                    )
            }
        };
        let (byte_encoder, byte_decoder) = gpt2_byte_tables();
        Ok(Self {
            model,
            tokens,
            token_to_id,
            merges,
            max_token_len,
            byte_to_id,
            byte_encoder,
            byte_decoder,
            bos,
            eos,
            add_bos,
        })
    }
}
