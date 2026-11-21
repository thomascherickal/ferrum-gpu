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

    /// The tokenizer family declared by the file (`tokenizer.ggml.model`).
    pub fn model(&self) -> TokModel {
        self.model
    }
    /// Number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    /// The beginning-of-sequence token id, if the file defines one.
    pub fn bos(&self) -> Option<usize> {
        self.bos.map(|x| x as usize)
    }
    /// The BOS id to prepend to a prompt: `bos()` only when the file asks for it
    /// (`add_bos_token`, defaulting like llama.cpp). Prepending an unwanted BOS changes
    /// the model's output (e.g. SmolLM2 has a BOS id but `add_bos_token = false`).
    pub fn prefix_bos(&self) -> Option<usize> {
        self.bos().filter(|_| self.add_bos)
    }
    /// The end-of-sequence token id, if the file defines one.
    pub fn eos(&self) -> Option<usize> {
        self.eos.map(|x| x as usize)
    }

    /// Exact-piece vocabulary lookup (used to map chat special tokens to ids).
    pub fn token_id(&self, piece: &str) -> Option<usize> {
        self.token_to_id.get(piece).map(|&id| id as usize)
    }

    /// Encode text to token IDs. BPE is exact (given the approximate
    /// pre-tokenizer); SPM is a greedy longest-match approximation with byte
    /// fallback. See the module docs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.model {
            TokModel::Bpe => self.encode_bpe(text),
            TokModel::Spm => self.encode_spm(text),
        }
    }

    /// Decode token IDs back to text. Exact for both families.
    pub fn decode(&self, ids: &[usize]) -> String {
        match self.model {
            TokModel::Bpe => self.decode_bpe(ids),
            TokModel::Spm => self.decode_spm(ids),
        }
    }

    // ── BPE ───────────────────────────────────────────────────────────────────

    pub(super) fn encode_bpe(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            // Byte-level map: each byte → one printable GPT-2 char (a "piece").
            let mut pieces: Vec<String> = chunk
                .bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect();
            // Greedy lowest-rank merges until none apply (standard BPE). `key` is
            // reused across every pair lookup so the scan allocates nothing.
            let mut key = String::new();
            while pieces.len() > 1 {
                let mut best: Option<(usize, u32)> = None;
                for i in 0..pieces.len() - 1 {
                    key.clear();
                    key.push_str(&pieces[i]);
                    key.push('\u{0}');
                    key.push_str(&pieces[i + 1]);
                    if let Some(&r) = self.merges.get(key.as_str()) {
                        let better = match best {
                            None => true,
                            Some((_, br)) => r < br,
                        };
                        if better {
                            best = Some((i, r));
                        }
                    }
                }
                let Some((i, _)) = best else { break };
                let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                pieces[i] = merged;
                pieces.remove(i + 1);
            }
            for p in pieces {
                if let Some(&id) = self.token_to_id.get(&p) {
                    out.push(id as usize);
                } else {
                    // Unmergeable leftover: emit its constituent byte tokens,
                    // which always exist in a byte-level vocab.
                    for ch in p.chars() {
                        if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                            out.push(id as usize);
                        }
                    }
                }
            }
        }
        out
    }

    pub(super) fn decode_bpe(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            for ch in tok.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    // A non-byte char (e.g. an added special token's literal
                    // text) — emit its own UTF-8.
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // ── SPM ───────────────────────────────────────────────────────────────────

    pub(super) fn encode_spm(&self, text: &str) -> Vec<usize> {
        // SentencePiece prepends a dummy space and renders spaces as ▁.
        let normalized: Vec<char> = format!(" {text}")
            .replace(' ', &SPM_SPACE.to_string())
            .chars()
            .collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < normalized.len() {
            // Longest token in the vocab starting at i. The window is capped at
            // the longest vocab token, so this is O(max_token_len) per position
            // (not O(remaining-length)) — without the cap, encoding a long corpus
            // is O(n³) and effectively hangs.
            let mut hit = None;
            let mut j = (i + self.max_token_len).min(normalized.len());
            while j > i {
                let cand: String = normalized[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&cand) {
                    hit = Some((id, j));
                    break;
                }
                j -= 1;
            }
            match hit {
                Some((id, j)) => {
                    out.push(id as usize);
                    i = j;
                }
                None => {
                    // Byte fallback: emit each UTF-8 byte's <0xXX> token.
                    let mut buf = [0u8; 4];
                    for &b in normalized[i].encode_utf8(&mut buf).as_bytes() {
                        if let Some(&id) = self.byte_to_id.get(&b) {
                            out.push(id as usize);
                        }
                    }
                    i += 1;
                }
            }
        }
        out
    }

    pub(super) fn decode_spm(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            if let Some(b) = parse_byte_token(tok) {
                bytes.push(b);
                continue;
            }
            for ch in tok.chars() {
                if ch == SPM_SPACE {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        // Drop the leading space SentencePiece's dummy prefix introduced.
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }
}
