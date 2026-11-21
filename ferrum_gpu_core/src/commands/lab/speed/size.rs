use super::*;

/// The two model sizes the lab can measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    /// 4 layers, width 256 (about 3 million weights): quick.
    Small,
    /// 8 layers, width 512 (about 26 million weights): closer to a real small model.
    Medium,
}

impl Size {
    pub fn parse(s: &str) -> Result<Size, String> {
        match s {
            "small" => Ok(Size::Small),
            "medium" => Ok(Size::Medium),
            o => Err(format!("--size: unknown '{o}' (small, medium)")),
        }
    }

    pub fn config(self) -> Config {
        let (vocab, dim, heads, head_dim, layers, hidden) = match self {
            Size::Small => (512, 256, 8, 32, 4, 704),
            Size::Medium => (2048, 512, 8, 64, 8, 1408),
        };
        Config {
            vocab,
            ctx: 64,
            dim,
            heads,
            kv_heads: 4,
            head_dim,
            layers,
            hidden,
            rope_dim: head_dim,
            rope_base: 10000.0,
            rope_type: RopeType::Norm,
            eps: 1e-5,
        }
    }

    /// The sentence the report prints about the model.
    pub fn describe(self) -> String {
        let c = self.config();
        format!(
            "random weights, {} layers, width {}, vocabulary {}, 32 tokens per run",
            c.layers, c.dim, c.vocab
        )
    }
}

/// Tokens per run of both measurements.
pub const TOKENS: usize = 32;

pub(super) fn ids(vocab: usize) -> Vec<usize> {
    (0..TOKENS).map(|i| (i * 7 + 3) % vocab).collect()
}

/// Tokens per second for `tokens` processed in `ms` milliseconds.
pub fn tps(tokens: usize, ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / (ms / 1e3)
    }
}

/// Run `f` `reps` times and return the mean milliseconds, or the first error it reported.
pub(super) fn time_fallible(
    reps: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<f64, String> {
    let mut fault: Option<String> = None;
    let ms = time_ms(reps, || {
        if let Err(e) = f() {
            fault.get_or_insert(e);
        }
    });
    fault.map_or(Ok(ms), Err)
}
