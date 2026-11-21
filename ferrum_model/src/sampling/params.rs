/// Sampling controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// `0.0` = greedy argmax.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens (`0` = off).
    pub top_k: usize,
    /// Keep the smallest set whose probability reaches `top_p` (`1.0` = off).
    pub top_p: f32,
    /// Divide positive (multiply negative) logits of recent tokens by this (`1.0` = off).
    pub repetition_penalty: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Index of the first maximum (NaN never wins).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] || logits[best].is_nan() && !v.is_nan() {
            best = i;
        }
    }
    best
}
