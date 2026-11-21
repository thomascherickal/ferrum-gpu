use super::*;

impl ByteBpeTokenizer {
    /// Rank-based BPE encode of one pre-tokenized chunk (K1): repeatedly apply
    /// the lowest-rank (earliest-learned) applicable merge until none applies —
    /// `O(applied merges × len)` instead of rescanning the whole text once per
    /// learned merge.
    pub(super) fn encode_chunk(&self, bytes: &[u8], out: &mut Vec<usize>) {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        while ids.len() >= 2 {
            // Lowest-rank adjacent pair present in the sequence.
            let mut best: Option<(usize, (u32, u32))> = None;
            for w in ids.windows(2) {
                let pair = (w[0], w[1]);
                if let Some(&rank) = self.ranks.get(&pair) {
                    if best.is_none_or(|(br, _)| rank < br) {
                        best = Some((rank, pair));
                    }
                }
            }
            let Some((rank, pair)) = best else { break };
            ids = merge_pair(&ids, pair, (BASE_VOCAB + rank) as u32);
        }
        out.extend(ids.into_iter().map(|i| i as usize));
    }

    /// Like [`encode`](Self::encode), but every [`DOC_SEP`] becomes the EOS id
    /// when this tokenizer has one. Without EOS it is exactly `encode`.
    pub fn encode_docs(&self, text: &str) -> Vec<usize> {
        let Some(eos) = self.eos_id() else {
            return self.encode(text);
        };
        let mut ids = Vec::new();
        for (i, doc) in text.split(DOC_SEP).enumerate() {
            if i > 0 {
                ids.push(eos);
            }
            ids.extend(self.encode(doc));
        }
        ids
    }

    /// Encode text to token IDs: pre-tokenize at whitespace boundaries (K2),
    /// then rank-based BPE-encode each chunk (K1) and concatenate.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            self.encode_chunk(chunk.as_bytes(), &mut out);
        }
        out
    }

    /// Encode `text`, optionally framed with the registered BOS/EOS tokens
    /// (I3/K3). Framing is skipped silently for any token not registered.
    pub fn encode_framed(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if bos {
            if let Some(b) = self.bos_id() {
                ids.push(b);
            }
        }
        ids.extend(self.encode(text));
        if eos {
            if let Some(e) = self.eos_id() {
                ids.push(e);
            }
        }
        ids
    }

    /// Decode token IDs back to text. Unknown IDs are skipped; byte sequences
    /// that are not valid UTF-8 decode lossily (U+FFFD). Special tokens map to
    /// empty byte strings, so they vanish from decoded text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.vocab.get(id) {
                bytes.extend_from_slice(tok);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
