//! Log-likelihoods the way ferrum-cpu computes them (lm-eval conventions): trailing whitespace
//! of the context moves to the continuation, an empty context is scored after the BOS token,
//! and long texts are scored in sliding windows so every target is scored once with at least
//! half a window of history.

use crate::decoder::Decoder;

/// A summed log-probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    /// Sum of log p(token | earlier tokens).
    pub logprob: f64,
    /// Tokens scored.
    pub n_tokens: usize,
    /// Every scored token was the argmax.
    pub greedy: bool,
}

const ZERO: Score = Score {
    logprob: 0.0,
    n_tokens: 0,
    greedy: true,
};

/// Sum log p(ids[i] | ids[..i]) for `i in from..ids.len()` (`from >= 1`) over windows of the
/// decoder's context.
pub fn token_logprobs(d: &dyn Decoder, ids: &[usize], from: usize) -> Result<Score, String> {
    let from = from.max(1);
    if from >= ids.len() {
        return Ok(ZERO);
    }
    let t = d.info().ctx;
    let half = (t / 2).max(1);
    let mut s = ZERO;
    let mut next = from;
    while next < ids.len() {
        let start = if next <= t { 0 } else { next - half };
        let end = (start + t + 1).min(ids.len());
        let per = d.score(&ids[start..end])?;
        for target in next..end {
            let (lp, greedy) = per[target - start - 1];
            s.logprob += f64::from(lp);
            s.n_tokens += 1;
            s.greedy &= greedy;
        }
        next = end;
    }
    Ok(s)
}

/// log p(`continuation` | `context`).
pub fn pair(d: &dyn Decoder, context: &str, continuation: &str) -> Result<Score, String> {
    let trimmed = context.trim_end();
    let (context, continuation) = if trimmed.len() < context.len() {
        (
            trimmed,
            format!("{}{continuation}", &context[trimmed.len()..]),
        )
    } else {
        (context, continuation.to_string())
    };
    if continuation.is_empty() {
        return Ok(ZERO);
    }
    let ctx_ids = if context.is_empty() {
        vec![d.bos()]
    } else {
        d.encode(context)?
    };
    let joint = if context.is_empty() {
        None
    } else {
        Some(d.encode(&format!("{context}{continuation}"))?)
    };
    let ids = match joint {
        Some(j) if j.len() > ctx_ids.len() && j.starts_with(&ctx_ids) => j,
        _ => [ctx_ids.clone(), d.encode(&continuation)?].concat(),
    };
    token_logprobs(d, &ids, ctx_ids.len())
}

/// log-likelihood of all of `text`, the first token conditioned on the BOS token.
pub fn rolling(d: &dyn Decoder, text: &str) -> Result<Score, String> {
    let ids = [vec![d.bos()], d.encode(text)?].concat();
    token_logprobs(d, &ids, 1)
}

#[cfg(test)]
mod tests;
