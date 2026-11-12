//! Dividing `run`'s flags between its two halves, `train` and `generate`. Pure: no I/O, no model.

use super::flags::{generate_flags, train_flags};
use crate::flags::Flag;

/// `run`'s arguments, divided up.
#[derive(Debug, PartialEq)]
pub struct Split {
    /// Tokens for `train` (without `--out`, which `run` derives from the model path).
    pub train: Vec<String>,
    /// Tokens for `generate` (without `--model`).
    pub generate: Vec<String>,
    /// The model file: trained there if missing, loaded if present.
    pub model: String,
}

fn find(table: &[Flag], name: &str) -> Option<Flag> {
    table.iter().find(|f| f.name == name).copied()
}

/// Split `--name value` / `--name=value` into the name and the tokens it occupies, taking the value
/// from `it` when it is a separate word.
fn take(
    raw: &str,
    flag: Flag,
    it: &mut std::slice::Iter<'_, String>,
) -> Result<(String, Vec<String>), String> {
    let (name, inline) = match raw.split_once('=') {
        Some((n, v)) if n.starts_with("--") => (n, Some(v.to_string())),
        _ => (raw, None),
    };
    let mut toks = vec![name.to_string()];
    if flag.value {
        let v = match inline {
            Some(v) => v,
            None => it
                .next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))?,
        };
        toks.push(v);
    } else if inline.is_some() {
        return Err(format!("{name} takes no value"));
    }
    Ok((name.to_string(), toks))
}

/// Divide `rest`: a flag goes to every command whose table lists it (`--device`, `--threads` and
/// `--seed` go to both), `--model` names the file, and anything unknown is an error.
pub fn split(rest: &[String]) -> Result<Split, String> {
    let (tf, gf) = (train_flags(), generate_flags());
    let (mut train, mut generate, mut model) = (Vec::new(), Vec::new(), None);
    let mut it = rest.iter();
    while let Some(raw) = it.next() {
        let name = raw.split_once('=').map_or(raw.as_str(), |(n, _)| n);
        let flag = find(&tf, name)
            .or_else(|| find(&gf, name))
            .ok_or_else(|| format!("unknown flag '{raw}'"))?;
        let (name, toks) = take(raw, flag, &mut it)?;
        match name.as_str() {
            "--out" => {
                return Err("run takes --model for the model file (it is trained there if it does not exist); --out is for `train`".into())
            }
            "--model" => {
                if model.replace(toks[1].clone()).is_some() {
                    return Err("--model given twice".into());
                }
            }
            _ => {
                if find(&tf, &name).is_some() {
                    train.extend(toks.clone());
                }
                if find(&gf, &name).is_some() {
                    generate.extend(toks);
                }
            }
        }
    }
    let model = model.ok_or("--model is required")?;
    Ok(Split {
        train,
        generate,
        model,
    })
}

#[cfg(test)]
mod tests;
