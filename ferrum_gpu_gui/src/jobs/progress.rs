//! Reading the CLI's `--verbose` progress lines into events.
//!
//! The formats are a contract with `ferrum_gpu_cli` (`cmds/gpt2/hooks.rs`):
//! `step 20/200 loss 3.1234 grad_norm 0.512 tokens/s 4800 elapsed 4.2s eta 38s` and
//! `epoch 3 val loss 2.9876 (best)`.

use crate::json::Json;

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
fn after(words: &[&str], key: &str) -> Option<f64> {
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

impl Event {
    /// The JSON the page reads.
    pub fn json(&self) -> Json {
        let n = |v: f64| Json::Num(v);
        match self {
            Event::Step {
                step,
                total,
                loss,
                grad_norm,
                tokens_per_s,
                elapsed,
                eta,
            } => Json::obj([
                ("kind", Json::Str("step".into())),
                ("step", n(*step as f64)),
                ("total", n(*total as f64)),
                ("loss", n(*loss)),
                ("grad_norm", n(*grad_norm)),
                ("tokens_per_s", n(*tokens_per_s)),
                ("elapsed", n(*elapsed)),
                ("eta", n(*eta)),
            ]),
            Event::Val { epoch, loss, best } => Json::obj([
                ("kind", Json::Str("val".into())),
                ("epoch", n(*epoch as f64)),
                ("loss", n(*loss)),
                ("best", Json::Bool(*best)),
            ]),
        }
    }
}
