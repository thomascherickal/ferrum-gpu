use super::*;

/// Feed `prompt`, then sample `opts.max_new` tokens (stopping when the context is full), using
/// `feed` to turn a token into the next-token logits. Returns only the new tokens.
pub fn generate_with(
    feed: &mut dyn FnMut(usize) -> Result<Vec<f32>, String>,
    ctx: usize,
    prompt: &[usize],
    opts: &GenOpts,
) -> Result<Vec<usize>, String> {
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    if prompt.len() >= ctx {
        return Err(format!(
            "the prompt has {} tokens but the context is {ctx}",
            prompt.len()
        ));
    }
    let params = Params {
        temperature: opts.temperature,
        top_k: opts.top_k,
        ..Params::default()
    };
    let mut rng = Rng::new(opts.seed);
    let mut logits = Vec::new();
    for &t in prompt {
        crate::cancel::check()?;
        logits = feed(t)?;
    }
    let mut history = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < opts.max_new && history.len() < ctx {
        crate::cancel::check()?;
        let next = sample(&logits, &params, &history, &mut rng);
        out.push(next);
        history.push(next);
        if history.len() < ctx && out.len() < opts.max_new {
            logits = feed(next)?;
        }
    }
    Ok(out)
}

/// Training extras a backend may or may not support (see [`Backend::gpt2_engine_with`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineOptions {
    /// Share the token embedding and the output head.
    pub tie: bool,
    /// FFN-hidden dropout rate (`0.0` is off).
    pub dropout: f32,
    /// The run's seed (dropout masks are a function of it and the step).
    pub seed: u64,
}

/// A one-token-at-a-time GPT-2 decoder: feed a token id, get the next-token logits.
pub type Gpt2Feed<'a> = Box<dyn FnMut(usize) -> Result<Vec<f32>, String> + 'a>;
