//! The decoder the routes talk to, so tests can use a stub and the binary a real model.

use ferrum_gpu_gui::engine::{Backend, GenOpts};
use ferrum_ref::llama::Llama;
use std::sync::{Arc, Mutex};

/// Facts reported by `/v1/info`.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// Device name.
    pub device: String,
    /// `cpu` or `gpu`.
    pub kind: String,
    /// Parameter count.
    pub params: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Context length.
    pub ctx: usize,
}

/// What the server needs from a model.
pub trait Decoder: Send + Sync {
    /// Static facts.
    fn info(&self) -> Info;
    /// Continue `prompt` with sampled tokens (only the new ones are returned).
    fn generate(&self, prompt: &[usize], opts: &GenOpts) -> Result<Vec<usize>, String>;
    /// For each `i < len-1`: log-probability of `ids[i+1]` given `ids[..=i]` and whether it was
    /// the argmax.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String>;
}

/// A model on a backend. Calls are serialised: concurrent device use is not safe on every driver.
pub struct ModelDecoder {
    backend: Arc<dyn Backend>,
    model: Llama,
    gate: Mutex<()>,
}

impl ModelDecoder {
    /// Wrap `model` and `backend`.
    pub fn new(backend: Arc<dyn Backend>, model: Llama) -> Self {
        ModelDecoder {
            backend,
            model,
            gate: Mutex::new(()),
        }
    }
}

impl Decoder for ModelDecoder {
    fn info(&self) -> Info {
        Info {
            device: self.backend.name(),
            kind: self.backend.kind().into(),
            params: self.model.num_params(),
            vocab: self.model.cfg.vocab,
            ctx: self.model.cfg.ctx,
        }
    }

    fn generate(&self, prompt: &[usize], opts: &GenOpts) -> Result<Vec<usize>, String> {
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.backend.generate(&self.model, prompt, opts)
    }

    // ponytail: scoring runs on the CPU oracle; use GpuLlama::score if it becomes the bottleneck.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let (v, logits) = (self.model.cfg.vocab, self.model.forward_logits(ids)?);
        Ok((0..ids.len().saturating_sub(1))
            .map(|i| {
                let row = &logits[i * v..(i + 1) * v];
                let max = row.iter().copied().fold(f32::MIN, f32::max);
                let lse = row
                    .iter()
                    .map(|&x| f64::from(x - max).exp())
                    .sum::<f64>()
                    .ln();
                let lp = f64::from(row[ids[i + 1]]) - f64::from(max) - lse;
                (lp as f32, row[ids[i + 1]] >= max)
            })
            .collect())
    }
}

/// A fixed-answer decoder for tests.
pub struct StubDecoder;

impl Decoder for StubDecoder {
    fn info(&self) -> Info {
        Info {
            device: "stub".into(),
            kind: "cpu".into(),
            params: 7,
            vocab: 256,
            ctx: 16,
        }
    }

    /// Echoes `max_new` copies of `b'!'`.
    fn generate(&self, _prompt: &[usize], opts: &GenOpts) -> Result<Vec<usize>, String> {
        Ok(vec![usize::from(b'!'); opts.max_new])
    }

    /// Every token has log-probability `-1`; a token is the argmax when it is even.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        Ok(ids[1..].iter().map(|t| (-1.0, t % 2 == 0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_gpu_gui::engine::{fresh_model, CpuBackend};

    #[test]
    fn model_decoder_scores_and_generates_on_cpu() {
        let d = ModelDecoder::new(
            Arc::new(CpuBackend::new(1)),
            fresh_model("tiny", 1).unwrap(),
        );
        let i = d.info();
        assert_eq!((i.kind.as_str(), i.vocab, i.ctx), ("cpu", 256, 64));
        let s = d.score(&[1, 2, 3, 4]).unwrap();
        assert_eq!(s.len(), 3);
        assert!(s.iter().all(|(lp, _)| *lp <= 0.0 && lp.is_finite()));
        let o = GenOpts {
            max_new: 4,
            temperature: 0.0,
            top_k: 0,
            seed: 1,
        };
        assert_eq!(d.generate(&[1, 2], &o).unwrap().len(), 4);
        assert!(d.score(&[]).is_err());
    }

    #[test]
    fn stub_is_deterministic() {
        let o = GenOpts {
            max_new: 3,
            temperature: 0.0,
            top_k: 0,
            seed: 1,
        };
        assert_eq!(StubDecoder.generate(&[1], &o).unwrap(), vec![33; 3]);
        assert_eq!(
            StubDecoder.score(&[9, 2, 3]).unwrap(),
            vec![(-1.0, true), (-1.0, false)]
        );
    }
}
