//! The GPT-2 path of the CLI: what ferrum-cpu's `train_transformer` does. A GPT-2 model with
//! learned positions, LayerNorm and GELU, a byte-level BPE tokenizer stored inside the GGUF, so
//! llama.cpp and ferrum-cpu can run the file.
//!
//! Sizes, defaults and flag spellings follow ferrum-cpu (docs/PARITY.md); the `llama` path is the
//! one used for imported Llama/Qwen files and `--arch llama`.

mod extras;
pub mod gen;
mod hooks;
pub mod prepare;
mod score;
mod tokenize;
mod train;

pub use gen::{eval, generate, is_gpt2, load, shape_of};
pub use tokenize::tokenize;
pub use train::train;

use crate::flags::{many, switch, val, Args, Flag};
use ferrum_model::gpt2_gguf::Quant;

/// A named starting size (ferrum-cpu's `--preset`).
pub struct Preset {
    /// Name given to `--preset`.
    pub name: &'static str,
    ctx: usize,
    dim: usize,
    heads: usize,
    layers: usize,
    hidden: usize,
    vocab: usize,
    lr: f32,
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

impl Sizes {
    /// ferrum-cpu's defaults (no preset): context 16, embed 32, 4 heads, 2 blocks, hidden 64,
    /// vocab 512, lr 0.01.
    pub const DEFAULT: Sizes = Sizes {
        ctx: 16,
        dim: 32,
        heads: 4,
        layers: 2,
        hidden: 64,
        vocab: 512,
        lr: 0.01,
    };

    /// Defaults, then the preset, then explicit flags (flag order does not matter).
    pub fn from_flags(a: &Args) -> Result<Sizes, String> {
        let mut s = Sizes::DEFAULT;
        if let Some(name) = a.get("--preset") {
            let p = PRESETS.iter().find(|p| p.name == name).ok_or_else(|| {
                let valid: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
                format!("unknown --preset '{name}' (valid: {})", valid.join(", "))
            })?;
            s = Sizes {
                ctx: p.ctx,
                dim: p.dim,
                heads: p.heads,
                layers: p.layers,
                hidden: p.hidden,
                vocab: p.vocab,
                lr: p.lr,
            };
        }
        s.ctx = a.num("--context", s.ctx)?;
        s.dim = a.num("--embed", s.dim)?;
        s.heads = a.num("--heads", s.heads)?;
        s.layers = a.num("--blocks", s.layers)?;
        s.hidden = a.num("--hidden", s.hidden)?;
        s.vocab = a.num("--vocab", s.vocab)?;
        Ok(s)
    }
}

/// `--quant`: ferrum-cpu's `int8`/`int4`/`f32` or our `q8_0`/`q4_0`.
pub fn quant(a: &Args, default: &str) -> Result<Quant, String> {
    match a.get("--quant").unwrap_or(default) {
        "f32" => Ok(Quant::F32),
        "q8_0" | "int8" => Ok(Quant::Q8_0),
        "q4_0" | "int4" => Ok(Quant::Q4_0),
        o => Err(format!(
            "--quant: unknown '{o}' (f32, int8/q8_0, int4/q4_0)"
        )),
    }
}

/// Model-size flags.
pub const SIZE_FLAGS: [Flag; 7] = [
    val("--preset"),
    val("--context"),
    val("--embed"),
    val("--heads"),
    val("--blocks"),
    val("--hidden"),
    val("--vocab"),
];

/// Training-only flags of the GPT-2 path (`--arch llama` refuses them).
pub const TRAIN_FLAGS: [Flag; 22] = [
    val("--epochs"),
    val("--tokens"),
    val("--weight_decay"),
    val("--dropout"),
    val("--stride"),
    switch("--cosine"),
    val("--warmup"),
    val("--clip"),
    switch("--tie"),
    switch("--no-qat"),
    val("--tokenizer"),
    val("--checkpoint"),
    val("--checkpoint-every"),
    val("--resume"),
    val("--val"),
    val("--patience"),
    val("--max-seconds"),
    val("--metrics-csv"),
    switch("--verbose"),
    switch("--debug"),
    switch("--sample"),
    switch("--force"),
];

/// The first flag in `a` that only the GPT-2 path understands, if any.
pub fn only_gpt2(a: &Args) -> Option<&'static str> {
    TRAIN_FLAGS
        .iter()
        .chain(SIZE_FLAGS.iter())
        .map(|f| f.name)
        .filter(|n| *n != "--preset" && *n != "--context")
        .find(|n| a.has(n))
}

/// Generation flags: ours and ferrum-cpu's spellings.
pub const GEN_FLAGS: [Flag; 9] = [
    val("--temp"),
    val("--gen-seed"),
    val("--max-tokens"),
    val("--chars"),
    val("--top-p"),
    val("--repetition-penalty"),
    many("--stop"),
    switch("--stream"),
    val("--ids"),
];

#[cfg(test)]
mod tests;
