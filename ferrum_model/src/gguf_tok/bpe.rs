use super::*;

impl GgufTokenizer {
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
}
