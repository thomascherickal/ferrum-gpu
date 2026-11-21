use super::*;

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

/// The precisions `--quant` takes, for the error message.
pub const QUANT_NAMES: &str = "f32, f16, int8/q8_0, int4/q4_0, fp4/mxfp4, q2/q2_k";

/// `--quant`: ferrum-cpu's `int8`/`int4`/`f32`, our `q8_0`/`q4_0`, and `f16`, `fp4` (MXFP4), `q2` (Q2_K).
pub fn quant(a: &Args, default: &str) -> Result<Quant, String> {
    let v = a.get("--quant").unwrap_or(default);
    Quant::parse(v).ok_or_else(|| format!("--quant: unknown '{v}' ({QUANT_NAMES})"))
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
