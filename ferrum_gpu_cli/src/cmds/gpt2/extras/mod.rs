//! The training options beyond the plain loop: `--tie` and `--dropout`.
//!
//! `--tie` runs on both engines (the GPU has the transpose kernel and the fold). `--dropout` runs
//! on the CPU trainer only: the GPU engine has no kernel to draw a dropout mask. This module
//! decides, in one place, what that means for a run: parse and validate the options, pick the
//! device (dropout with an explicit `--device gpu` is refused by name, `auto` quietly uses the
//! CPU), and build the matching engine.

use crate::cmds::{app, app_requested, say};
use crate::device::Workload;
use crate::flags::Args;
use ferrum_gpu_gui::app::{App, DeviceChoice};
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
    /// Does this run need the CPU engine? (Only dropout does; a tied head trains on either.)
    pub fn needs_cpu(&self) -> bool {
        self.dropout > 0.0
    }

    /// Does the run use an option the plain engine constructor does not take?
    fn any(&self) -> bool {
        self.tie || self.dropout > 0.0
    }

    /// The options as the flags spelled them, for messages.
    fn names(&self) -> String {
        match (self.tie, self.dropout > 0.0) {
            (true, true) => "--tie and --dropout".into(),
            (true, false) => "--tie".into(),
            _ => "--dropout".into(),
        }
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

/// The app for a run: the usual device choice, except that [`Extras`] force the CPU engine.
pub fn app_for(a: &Args, x: &Extras, out: &mut dyn Write) -> Result<App, String> {
    if !x.needs_cpu() {
        return app(a, Workload::Train);
    }
    // ponytail: dropout is CPU only. Upgrade: a mask kernel on the GPU (the mask is a pure function
    // of (seed, block, index), but a u64 hash; the kernel needs a 32-bit one on both sides).
    if DeviceChoice::parse(a.get("--device").unwrap_or("auto"))? == DeviceChoice::Gpu {
        return Err(
            "--dropout trains on the CPU engine only for now; use --device cpu (or auto) or drop it"
                .into(),
        );
    }
    say(out, format!("{}: training on the CPU engine", x.names()))?;
    if x.dropout > MAX_RATE {
        say(
            out,
            format!("--dropout {} is used as {MAX_RATE}", x.dropout),
        )?;
    }
    app_requested(a, Workload::Train, DeviceChoice::Cpu)
}

/// The training engine for `start`: the backend's ordinary one, or the CPU one with [`Extras`].
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
