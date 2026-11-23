use super::*;

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
        let _g = crate::sync::lock(&self.gate);
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
            stop_ids: vec![],
        };
        run(
            &mut *feed,
            &|ids| self.tok.decode(ids),
            self.model.cfg.ctx,
            &self.tok.encode_docs(prompt),
            &o,
            None,
        )
    }

    // ponytail: scoring runs on the CPU oracle, like the Llama path.
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        let _g = crate::sync::lock(&self.gate);
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
