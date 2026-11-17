//! Speed per precision on a random-weight model: decode (one token at a time with the KV cache) and
//! prefill (a whole sequence at once). Random weights give the same speed as trained ones, but not
//! text, and the model is small, so this measures launch and dispatch cost more than bandwidth;
//! the report says so.

use super::report::Speed;
use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_wgpu::bench_util::time_ms;
use ferrum_wgpu::llama::{plan, GpuLlama};
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::Gpu;

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

fn ids(vocab: usize) -> Vec<usize> {
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

/// Measure one precision (`None` is f32).
pub fn measure(gpu: &Gpu, w: &Llama, quant: Option<QKind>, reps: usize) -> Result<Speed, String> {
    let c = w.cfg;
    let ids = ids(c.vocab);
    let mut m = GpuLlama::new(gpu, w, quant, None).map_err(|e| e.to_string())?;
    let mut fault: Option<String> = None;
    let decode_ms = time_ms(reps, || {
        m.reset();
        for &t in &ids {
            if let Err(e) = m.feed(t) {
                fault.get_or_insert(e.to_string());
            }
        }
    });
    let prefill_ms = time_ms(reps, || {
        if let Err(e) = m.forward_all(&ids) {
            fault.get_or_insert(e.to_string());
        }
    });
    if let Some(e) = fault {
        return Err(e);
    }
    Ok(Speed {
        precision: quant
            .map_or("f32", |q| match q {
                QKind::Q8_0 => "Q8_0",
                QKind::Q4_0 => "Q4_0",
            })
            .into(),
        model_mib: plan(&c, quant).total() as f64 / 1048576.0,
        decode_tps: tps(TOKENS, decode_ms),
        prefill_tps: tps(TOKENS, prefill_ms),
    })
}

/// The CPU reference's prefill speed on the same model, for scale.
pub fn cpu_prefill(w: &Llama, reps: usize) -> Result<f64, String> {
    let ids = ids(w.cfg.vocab);
    let mut fault: Option<String> = None;
    let ms = time_ms(reps, || {
        if let Err(e) = w.forward_logits(&ids) {
            fault.get_or_insert(e);
        }
    });
    fault.map_or(Ok(tps(TOKENS, ms)), Err)
}

/// All three precisions.
pub fn run(gpu: &Gpu, size: Size, reps: usize) -> Result<(Vec<Speed>, f64), String> {
    let w = Llama::random(size.config(), 5, false, false);
    let mut speeds = Vec::new();
    for q in [None, Some(QKind::Q8_0), Some(QKind::Q4_0)] {
        speeds.push(measure(gpu, &w, q, reps)?);
    }
    Ok((speeds, cpu_prefill(&w, reps)?))
}

#[cfg(test)]
mod tests;
