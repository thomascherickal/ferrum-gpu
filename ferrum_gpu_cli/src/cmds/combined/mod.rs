//! `run`: ferrum-cpu's train-then-generate in one command, flag-style.
//!
//! `run --data F --model M.gguf --prompt TEXT [train flags] [generate flags]` trains `M.gguf` on `F`
//! when the file does not exist yet (loading it as is when it does, like ferrum-cpu), then generates
//! from the prompt. Every flag goes to the command it belongs to ([`split`]); the model file is
//! `--model`, not `--out`.

mod flags;
mod split;

use super::{dispatch, say};
use split::{split, Split};
use std::io::Write;

/// Can the generation half run? Checked before any training, so a typo does not cost a training run.
fn check_generation_half(generate: &[String]) -> Result<(), String> {
    if generate.iter().any(|t| t == "--prompt" || t == "--ids") {
        Ok(())
    } else {
        Err("run needs --prompt (or --ids) for the generation half".into())
    }
}

/// The `train` invocation for a model that does not exist yet.
fn train_command(s: &Split) -> Vec<String> {
    let mut t = vec!["train".to_string()];
    t.extend(s.train.iter().cloned());
    t.extend(["--out".to_string(), s.model.clone()]);
    t
}

/// The `generate` invocation on the (now existing) model.
fn generate_command(s: &Split) -> Vec<String> {
    let mut g = vec![
        "generate".to_string(),
        "--model".to_string(),
        s.model.clone(),
    ];
    g.extend(s.generate.iter().cloned());
    g
}

/// `run`: train `--model` on `--data` if the file is missing, then generate.
pub fn run_combined(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let s = split(rest)?;
    check_generation_half(&s.generate)?;
    if std::path::Path::new(&s.model).exists() {
        say(
            out,
            format!("{} exists: loading it (training skipped)", s.model),
        )?;
    } else {
        dispatch(&train_command(&s), out)?;
    }
    dispatch(&generate_command(&s), out)
}

#[cfg(test)]
mod tests;
