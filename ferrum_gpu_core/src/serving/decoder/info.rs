use super::*;

/// Facts reported by `/v1/info`.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// Device name.
    pub device: String,
    /// `cpu` or `gpu`.
    pub kind: String,
    /// `gpt2` or `llama`.
    pub arch: String,
    /// Parameter count.
    pub params: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Context length.
    pub ctx: usize,
}

/// One generate request, after validation.
#[derive(Clone, Debug, PartialEq)]
pub struct GenReq {
    /// Tokens to add at most.
    pub max_new: usize,
    /// `0.0` is greedy.
    pub temperature: f32,
    /// `0` is off.
    pub top_k: usize,
    /// `1.0` is off.
    pub top_p: f32,
    /// `1.0` is off.
    pub repetition_penalty: f32,
    /// Sampler seed.
    pub seed: u64,
    /// Stop strings.
    pub stops: Vec<String>,
}

/// What the server needs from a model.
pub trait Decoder: Send + Sync {
    /// Static facts.
    fn info(&self) -> Info;
    /// Token ids of `text` (document-aware: `<|endoftext|>` is the end-of-text token).
    fn encode(&self, text: &str) -> Result<Vec<usize>, String>;
    /// The id scored before the first token of a text (end-of-text, or 0).
    fn bos(&self) -> usize;
    /// Continue `prompt`; returns the new text (cut before the first stop string) and the number
    /// of tokens sampled.
    fn complete(&self, prompt: &str, req: &GenReq) -> Result<(String, usize), String>;
    /// For each `i < len-1`: log-probability of `ids[i+1]` given `ids[..=i]` and whether it was
    /// the argmax. `ids` must fit the context.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String>;
}

/// Per-token `(log-probability, was argmax)` of `ids[1..]` from `[len, vocab]` logits.
pub fn logprobs(logits: &[f32], vocab: usize, ids: &[usize]) -> Vec<(f32, bool)> {
    (0..ids.len().saturating_sub(1))
        .map(|i| {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::MIN, f32::max);
            let lse = row
                .iter()
                .map(|&x| f64::from(x - max).exp())
                .sum::<f64>()
                .ln();
            let lp = f64::from(row[ids[i + 1]]) - f64::from(max) - lse;
            (lp as f32, row[ids[i + 1]] >= max)
        })
        .collect()
}

/// A Llama-style byte-level model on a backend. Calls are serialised: concurrent device use is
/// not safe on every driver.
pub struct ModelDecoder {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) model: Llama,
    pub(crate) gate: Mutex<()>,
}
