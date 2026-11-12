//! The training options beyond the plain loop: `--tie` and `--dropout`.
//!
//! Both run on both engines, so the device is chosen as for any other training run. This module
//! parses and validates the options, says so when a rate is clamped, and builds the matching
//! engine (the backend's plain one when neither option is given).

use crate::cmds::{app, say};
use crate::device::Workload;
use crate::flags::Args;
use ferrum_gpu_gui::app::App;
use ferrum_gpu_gui::engine::{Engine, EngineOptions};
use ferrum_ref::dropout::MAX_RATE;
use ferrum_ref::gpt2::Gpt2;
use std::io::Write;

/// The validated `--tie` / `--dropout` options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extras {
    /// Share the embedding and the output head.
    pub tie: bool,
    /// FFN-hidden dropout rate (`0.0` is off).
    pub dropout: f32,
}

impl Extras {
    /// Does the run use an option the plain engine constructor does not take?
    fn any(&self) -> bool {
        self.tie || self.dropout > 0.0
    }
}

/// Parse `--tie` and `--dropout`; a rate must be in `[0, 1)`, and one above [`MAX_RATE`] is used as
/// that (ferrum-cpu clamps the same way).
pub fn parse(a: &Args) -> Result<Extras, String> {
    let dropout: f32 = a.num("--dropout", 0.0)?;
    if !(0.0..1.0).contains(&dropout) {
        return Err(format!("--dropout must be in [0, 1), got {dropout}"));
    }
    Ok(Extras {
        tie: a.has("--tie"),
        dropout,
    })
}

/// The app for a run: the usual device choice. A rate above the maximum is announced as clamped.
pub fn app_for(a: &Args, x: &Extras, out: &mut dyn Write) -> Result<App, String> {
    if x.dropout > MAX_RATE {
        say(
            out,
            format!("--dropout {} is used as {MAX_RATE}", x.dropout),
        )?;
    }
    app(a, Workload::Train)
}

/// The training engine for `start`: the backend's plain one, or the one with [`Extras`].
pub fn engine<'a>(
    app: &'a App,
    start: &Gpt2,
    qat: bool,
    x: &Extras,
    seed: u64,
) -> Result<Box<dyn Engine + 'a>, String> {
    if x.any() {
        app.backend.gpt2_engine_with(
            start,
            qat,
            &EngineOptions {
                tie: x.tie,
                dropout: x.dropout,
                seed,
            },
        )
    } else {
        app.backend.gpt2_engine(start, qat)
    }
}

#[cfg(test)]
mod tests;
