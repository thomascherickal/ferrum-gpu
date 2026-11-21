use super::*;

impl GgufTokenizer {
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
}
