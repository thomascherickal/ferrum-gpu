use super::*;

impl GgufTokenizer {
    // ── SPM ───────────────────────────────────────────────────────────────────

    pub(crate) fn encode_spm(&self, text: &str) -> Vec<usize> {
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

    pub(crate) fn decode_spm(&self, ids: &[usize]) -> String {
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
