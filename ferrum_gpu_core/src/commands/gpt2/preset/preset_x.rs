/// A named starting size (ferrum-cpu's `--preset`).
pub struct Preset {
    /// Name given to `--preset`.
    pub name: &'static str,
    pub(crate) ctx: usize,
    pub(crate) dim: usize,
    pub(crate) heads: usize,
    pub(crate) layers: usize,
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    pub(crate) lr: f32,
}

/// `tiny` ~0.2M parameters, `small` ~1.5M, `medium` ~7M.
pub const PRESETS: [Preset; 3] = [
    Preset {
        name: "tiny",
        ctx: 64,
        dim: 64,
        heads: 4,
        layers: 2,
        hidden: 256,
        vocab: 512,
        lr: 1e-2,
    },
    Preset {
        name: "small",
        ctx: 128,
        dim: 128,
        heads: 4,
        layers: 4,
        hidden: 512,
        vocab: 2048,
        lr: 3e-3,
    },
    Preset {
        name: "medium",
        ctx: 256,
        dim: 256,
        heads: 8,
        layers: 6,
        hidden: 1024,
        vocab: 4096,
        lr: 1e-3,
    },
];

/// Model sizes after presets and explicit flags.
#[derive(Clone, Debug, PartialEq)]
pub struct Sizes {
    /// Context length.
    pub ctx: usize,
    /// Embedding width.
    pub dim: usize,
    /// Attention heads.
    pub heads: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// Feed-forward width.
    pub hidden: usize,
    /// Vocabulary size to train the tokenizer to.
    pub vocab: usize,
    /// Default learning rate.
    pub lr: f32,
}
