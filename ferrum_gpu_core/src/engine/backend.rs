use super::*;

/// A place to run models.
pub trait Backend: Send + Sync {
    /// A GPT-2 training engine for `w` (quantisation-aware when `qat`), borrowed from the backend.
    fn gpt2_engine<'a>(&'a self, _w: &Gpt2, _qat: bool) -> Result<Box<dyn Engine + 'a>, String> {
        Err(format!(
            "GPT-2 training is not available on {}",
            self.name()
        ))
    }

    /// Like [`Backend::gpt2_engine`], with the extras in `opts` (a tied head, dropout).
    fn gpt2_engine_with<'a>(
        &'a self,
        _w: &Gpt2,
        _qat: bool,
        _opts: &EngineOptions,
    ) -> Result<Box<dyn Engine + 'a>, String> {
        Err(format!(
            "tied heads and dropout are not available on {}",
            self.name()
        ))
    }

    /// A GPT-2 decoder with a fresh KV cache.
    fn gpt2_decoder<'a>(&'a self, _w: &Gpt2) -> Result<Gpt2Feed<'a>, String> {
        Err(format!(
            "GPT-2 decoding is not available on {}",
            self.name()
        ))
    }

    /// A Llama decoder with a fresh KV cache (token in, next-token logits out).
    fn llama_decoder<'a>(&'a self, _m: &'a Llama) -> Result<Gpt2Feed<'a>, String> {
        Err(format!("decoding is not available on {}", self.name()))
    }

    /// `"cpu"` or `"gpu"`.
    fn kind(&self) -> &'static str;
    /// Human-readable device name.
    fn name(&self) -> String;
    /// Train `model` on `spec.text`, updating `progress`; returns the trained weights.
    fn train(
        &self,
        model: &Llama,
        spec: &TrainSpec,
        progress: &Arc<Mutex<Progress>>,
    ) -> Result<Llama, String>;
    /// Continue `prompt` with sampled tokens.
    fn generate(
        &self,
        model: &Llama,
        prompt: &[usize],
        opts: &GenOpts,
    ) -> Result<Vec<usize>, String>;
}
