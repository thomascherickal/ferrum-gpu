use super::*;

/// All weights of a GPT-2 model.
#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2 {
    /// Hyperparameters.
    pub cfg: Config,
    /// Token embeddings `[vocab, dim]` (one row per token).
    pub tok: Vec<f32>,
    /// Position embeddings `[ctx, dim]`.
    pub pos: Vec<f32>,
    /// Blocks in order.
    pub blocks: Vec<Block>,
    /// Final LayerNorm gain.
    pub lnf_g: Vec<f32>,
    /// Final LayerNorm shift.
    pub lnf_b: Vec<f32>,
    /// LM head `[dim, vocab]`, no bias.
    pub head: Vec<f32>,
}
