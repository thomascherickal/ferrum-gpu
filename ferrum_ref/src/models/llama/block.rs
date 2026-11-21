use super::*;

/// One transformer block's parameters. A bias vector is empty when the model has none (Llama has
/// no biases; Qwen2 has them on q, k and v only).
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Attention RMSNorm gain.
    pub attn_norm: Vec<f32>,
    /// Query projection `[dim, q_dim]`.
    pub wq: Vec<f32>,
    /// Query bias (empty = none).
    pub bq: Vec<f32>,
    /// Key projection `[dim, kv_dim]`.
    pub wk: Vec<f32>,
    /// Key bias (empty = none).
    pub bk: Vec<f32>,
    /// Value projection `[dim, kv_dim]`.
    pub wv: Vec<f32>,
    /// Value bias (empty = none).
    pub bv: Vec<f32>,
    /// Output projection `[q_dim, dim]`.
    pub wo: Vec<f32>,
    /// FFN RMSNorm gain.
    pub ffn_norm: Vec<f32>,
    /// SwiGLU gate projection `[dim, hidden]`.
    pub gate: Vec<f32>,
    /// SwiGLU up projection `[dim, hidden]`.
    pub up: Vec<f32>,
    /// Down projection `[hidden, dim]`.
    pub down: Vec<f32>,
}

/// All weights of a Llama-family model.
#[derive(Clone, Debug, PartialEq)]
pub struct Llama {
    /// Hyperparameters.
    pub cfg: Config,
    /// RoPE frequency scaling.
    pub rope: RopeScaling,
    /// Token embeddings `[vocab, dim]`.
    pub tok: Vec<f32>,
    /// Blocks in order.
    pub blocks: Vec<Block>,
    /// Final RMSNorm gain.
    pub out_norm: Vec<f32>,
    /// LM head `[dim, vocab]`, no bias. When `tied` it is the transposed embedding.
    pub head: Vec<f32>,
    /// Whether the head is tied to the embedding (the GGUF omits `output.weight`).
    pub tied: bool,
}

/// Per-layer key/value history for cached decoding (`[pos, kv_dim]` each).
#[derive(Clone, Debug, Default)]
pub struct Cache {
    pub(crate) k: Vec<Vec<f32>>,
    pub(crate) v: Vec<Vec<f32>>,
}

impl Cache {
    /// An empty cache for a model with `layers` blocks.
    pub fn new(layers: usize) -> Self {
        Cache {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
        }
    }

    /// Tokens consumed so far, for a key/value width of `kv_dim`.
    pub fn len(&self, kv_dim: usize) -> usize {
        self.k.first().map_or(0, |k| k.len() / kv_dim)
    }

    /// True when no token has been consumed.
    pub fn is_empty(&self) -> bool {
        self.k.first().is_none_or(|k| k.is_empty())
    }
}
