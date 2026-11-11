//! The decoders the routes talk to: a stub for tests, a Llama-style byte model, and a GPT-2
//! model with its BPE tokenizer.

use ferrum_gpu_cli::cmds::gpt2::gen::{run, Loaded, Opts};
use ferrum_gpu_gui::engine::{Backend, GenOpts};
use ferrum_gpu_gui::tokenizer::{decode as bytes_to_text, encode as text_to_bytes};
use ferrum_model::sampling::Params;
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::llama::Llama;
use ferrum_tok::ByteBpeTokenizer;
use std::sync::{Arc, Mutex};

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

/// Cut `text` before the first of `stops`.
pub fn cut_at_stop(mut text: String, stops: &[String]) -> String {
    if let Some(at) = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        text.truncate(at);
    }
    text
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
            arch: "llama".into(),
            params: self.model.num_params(),
            vocab: self.model.cfg.vocab,
            ctx: self.model.cfg.ctx,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        Ok(text_to_bytes(text))
    }

    fn bos(&self) -> usize {
        0
    }

    fn complete(&self, prompt: &str, r: &GenReq) -> Result<(String, usize), String> {
        if r.top_p != 1.0 || r.repetition_penalty != 1.0 {
            return Err("top_p and repetition_penalty are only supported for GPT-2 models".into());
        }
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let opts = GenOpts {
            max_new: r.max_new,
            temperature: r.temperature,
            top_k: r.top_k,
            seed: r.seed,
        };
        let new = self
            .backend
            .generate(&self.model, &text_to_bytes(prompt), &opts)?;
        Ok((cut_at_stop(bytes_to_text(&new)?, &r.stops), new.len()))
    }

    // ponytail: scoring runs on the CPU oracle; use GpuLlama::score if it becomes the bottleneck.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        Ok(logprobs(
            &self.model.forward_logits(ids)?,
            self.model.cfg.vocab,
            ids,
        ))
    }
}

/// A GPT-2 model with its BPE tokenizer on a backend.
pub struct Gpt2Decoder {
    backend: Arc<dyn Backend>,
    model: Gpt2,
    tok: ByteBpeTokenizer,
    gate: Mutex<()>,
}

impl Gpt2Decoder {
    /// Wrap a loaded model (weights and tokenizer) and `backend`.
    pub fn new(backend: Arc<dyn Backend>, loaded: Loaded) -> Self {
        Gpt2Decoder {
            backend,
            model: loaded.model,
            tok: loaded.tok,
            gate: Mutex::new(()),
        }
    }
}

impl Decoder for Gpt2Decoder {
    fn info(&self) -> Info {
        let c = &self.model.cfg;
        Info {
            device: self.backend.name(),
            kind: self.backend.kind().into(),
            arch: "gpt2".into(),
            params: self.model.num_params(),
            vocab: c.vocab,
            ctx: c.ctx,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        Ok(self.tok.encode_docs(text))
    }

    fn bos(&self) -> usize {
        self.tok.eos_id().unwrap_or(0)
    }

    fn complete(&self, prompt: &str, r: &GenReq) -> Result<(String, usize), String> {
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let mut feed = self.backend.gpt2_decoder(&self.model)?;
        let o = Opts {
            max_new: r.max_new,
            params: Params {
                temperature: r.temperature,
                top_k: r.top_k,
                top_p: r.top_p,
                repetition_penalty: r.repetition_penalty,
            },
            seed: r.seed,
            stops: r.stops.clone(),
            stream: false,
        };
        run(
            &mut *feed,
            &self.tok,
            self.model.cfg.ctx,
            &self.tok.encode_docs(prompt),
            &o,
            &mut |_| {},
        )
    }

    // ponytail: scoring runs on the CPU oracle, like the Llama path.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        Ok(logprobs(
            &self.model.forward_logits(ids)?,
            self.model.cfg.vocab,
            ids,
        ))
    }
}

/// A fixed-answer decoder for tests: bytes as tokens, context 16.
pub struct StubDecoder;

impl Decoder for StubDecoder {
    fn info(&self) -> Info {
        Info {
            device: "stub".into(),
            kind: "cpu".into(),
            arch: "stub".into(),
            params: 7,
            vocab: 256,
            ctx: 16,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        Ok(text_to_bytes(text))
    }

    fn bos(&self) -> usize {
        0
    }

    /// `max_new` copies of `!`, cut at the first stop string.
    fn complete(&self, _prompt: &str, r: &GenReq) -> Result<(String, usize), String> {
        Ok((cut_at_stop("!".repeat(r.max_new), &r.stops), r.max_new))
    }

    /// Every token has log-probability `-1`; a token is the argmax when it is even.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        Ok(ids[1..].iter().map(|t| (-1.0, t % 2 == 0)).collect())
    }
}

#[cfg(test)]
mod tests;
