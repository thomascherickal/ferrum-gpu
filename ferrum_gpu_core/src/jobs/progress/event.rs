/// One parsed progress line.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A training step report.
    Step {
        /// Steps done.
        step: u64,
        /// Steps in the run.
        total: u64,
        /// Mean loss of the batch.
        loss: f64,
        /// Global gradient norm before clipping.
        grad_norm: f64,
        /// Training throughput.
        tokens_per_s: f64,
        /// Seconds since the run started.
        elapsed: f64,
        /// Estimated seconds left.
        eta: f64,
    },
    /// The result of `eval`: the mean loss over some tokens and its perplexity.
    Eval {
        /// Tokens predicted.
        tokens: u64,
        /// Mean cross-entropy.
        loss: f64,
        /// `exp(loss)`.
        perplexity: f64,
        /// Bits per byte of text, when the command reports it (models with their own tokenizer).
        bits_per_byte: Option<f64>,
    },
    /// The validation loss after an epoch.
    Val {
        /// Epochs done.
        epoch: u64,
        /// Mean loss on the held-out text.
        loss: f64,
        /// It improved on every earlier epoch.
        best: bool,
    },
}

/// The number after `key` in a whitespace-separated line, with an optional unit suffix removed.
pub(super) fn after(words: &[&str], key: &str) -> Option<f64> {
    let at = words.iter().position(|w| *w == key)?;
    let w = words.get(at + 1)?;
    w.trim_end_matches('s').parse().ok()
}

/// Parse one output line; `None` for anything that is not a progress line.
pub fn parse_line(line: &str) -> Option<Event> {
    let words: Vec<&str> = line.split_whitespace().collect();
    match words.first().copied()? {
        "step" => {
            let (step, total) = words.get(1)?.split_once('/')?;
            Some(Event::Step {
                step: step.parse().ok()?,
                total: total.parse().ok()?,
                loss: after(&words, "loss")?,
                grad_norm: after(&words, "grad_norm")?,
                tokens_per_s: after(&words, "tokens/s")?,
                elapsed: after(&words, "elapsed")?,
                eta: after(&words, "eta")?,
            })
        }
        "tokens:" if words.get(2) == Some(&"loss:") && words.get(4) == Some(&"perplexity:") => {
            Some(Event::Eval {
                tokens: words.get(1)?.parse().ok()?,
                loss: words.get(3)?.parse().ok()?,
                perplexity: words.get(5)?.parse().ok()?,
                bits_per_byte: (words.get(6) == Some(&"bits/byte:"))
                    .then(|| words.get(7)?.parse().ok())
                    .flatten(),
            })
        }
        "epoch" if words.get(2) == Some(&"val") && words.get(3) == Some(&"loss") => {
            Some(Event::Val {
                epoch: words.get(1)?.parse().ok()?,
                loss: words.get(4)?.parse().ok()?,
                best: words.get(5) == Some(&"(best)"),
            })
        }
        _ => None,
    }
}
