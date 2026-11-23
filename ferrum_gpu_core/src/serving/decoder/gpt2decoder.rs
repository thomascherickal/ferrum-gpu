use super::*;

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
        let _g = crate::sync::lock(&self.gate);
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
        let _g = crate::sync::lock(&self.gate);
        Ok(logprobs(
            &self.model.forward_logits(ids)?,
            self.model.cfg.vocab,
            ids,
        ))
    }
}

/// A GPT-2 model with its BPE tokenizer on a backend.
pub struct Gpt2Decoder {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) model: Gpt2,
    pub(crate) tok: ByteBpeTokenizer,
    pub(crate) gate: Mutex<()>,
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
