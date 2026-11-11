//! `generate`, `run-gguf` and `eval` for imported Llama/Qwen files that carry their own tokenizer
//! (`tokenizer.ggml.*`). Files without one are the byte-level models of `cmds::generate`.
//!
//! An instruct model is prompted through its chat template (ChatML, Llama-3, `[INST]`, Zephyr),
//! unless `--raw` or `--ids` is given; the reply ends at the end-of-turn token. In chat mode only
//! the reply is printed; in raw mode the prompt is printed first, as for GPT-2 files.

use super::gpt2::gen::{opts, run};
use super::{app, device_line, gate, mean_loss_ids, mem_mb, say, shape};
use crate::device::Workload;
use crate::flags::Args;
use crate::preflight;
use ferrum_gguf::Gguf;
use ferrum_model::chat_format::{detect, encode_segments, render, stop_token, ChatFormat};
use ferrum_model::gguf_tok::GgufTokenizer;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::Llama;
use std::io::Write;

/// A parsed imported model with its tokenizer and (if it is an instruct model) chat format.
pub struct Loaded {
    /// The weights.
    pub model: Llama,
    /// The file's own tokenizer.
    pub tok: GgufTokenizer,
    /// The chat format, when the file is recognised as an instruct model.
    pub chat: Option<ChatFormat>,
}

fn parse(path: &str) -> Result<Gguf, String> {
    Gguf::parse(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
        .map_err(|e| format!("{path}: {e}"))
}

/// Does the file carry a vocabulary? (`false` for unreadable files; the real error comes later.)
// ponytail: reads the whole file; a header-only probe (`Gguf::parse_prefix`) would avoid a second
// read of a large model.
pub fn has_tokenizer(path: &str) -> bool {
    parse(path).is_ok_and(|g| g.get("tokenizer.ggml.tokens").is_some())
}

/// Load the weights, the tokenizer and the chat format.
pub fn load(path: &str) -> Result<Loaded, String> {
    let g = parse(path)?;
    let model = from_gguf(&g).map_err(|e| format!("{path}: {e}"))?;
    let tok = GgufTokenizer::from_gguf(&g).map_err(|e| format!("{path}: {e}"))?;
    if tok.vocab_size() > model.cfg.vocab {
        return Err(format!(
            "{path}: the tokenizer has {} tokens but the model only {}",
            tok.vocab_size(),
            model.cfg.vocab
        ));
    }
    let arch = g.string("general.architecture").ok();
    let template = g.string("tokenizer.chat_template").ok();
    let chat = detect(template, arch, |p| tok.token_id(p).is_some());
    Ok(Loaded { model, tok, chat })
}

/// The prompt ids and whether they were wrapped in the chat template.
fn prompt_ids(l: &Loaded, a: &Args) -> Result<(Vec<usize>, bool), String> {
    if let Some(s) = a.get("--ids") {
        let ids = s
            .split_whitespace()
            .map(|x| {
                x.parse::<usize>()
                    .map_err(|_| format!("--ids: '{x}' is not an id"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((ids, false));
    }
    let text = a.need("--prompt")?;
    let mut ids: Vec<usize> = l.tok.prefix_bos().into_iter().collect();
    match l.chat.filter(|_| !a.has("--raw")) {
        Some(f) => {
            ids.extend(encode_segments(&l.tok, &render(f, text)));
            Ok((ids, true))
        }
        None => {
            ids.extend(l.tok.encode(text));
            Ok((ids, false))
        }
    }
}

/// The text of `new`, decoded after the token `anchor` so a leading space survives
/// SentencePiece's dummy-prefix stripping.
fn continuation(tok: &GgufTokenizer, anchor: usize, new: &[usize]) -> String {
    let mut ids = vec![anchor];
    ids.extend_from_slice(new);
    let full = tok.decode(&ids);
    let base = tok.decode(&[anchor]);
    full.strip_prefix(&base)
        .map_or(full.clone(), str::to_string)
}

/// `generate` / `run-gguf` for a model with its own tokenizer.
pub fn generate(a: &Args, out: &mut dyn Write, verbose: bool) -> Result<(), String> {
    let l = load(a.need("--model")?)?;
    let mut o = opts(a)?;
    let (prompt, chat) = prompt_ids(&l, a)?;
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    o.stop_ids = l.tok.eos().into_iter().collect();
    if let (true, Some(f)) = (chat, l.chat) {
        o.stop_ids.extend(l.tok.token_id(stop_token(f)));
    }
    let app = app(a, Workload::Decode)?;
    if verbose {
        device_line(out, &app)?;
        say(out, shape(&l.model))?;
    }
    gate(out, &app, &preflight::decode_buffers(&l.model), mem_mb(a)?)?;
    let mut feed = app.backend.llama_decoder(&l.model)?;
    let anchor = prompt[prompt.len() - 1];
    let shown = if chat {
        String::new()
    } else {
        a.get("--prompt").unwrap_or("").to_string()
    };
    if o.stream {
        write!(out, "{shown}").map_err(|e| e.to_string())?;
    }
    let mut werr = None;
    let (text, _) = run(
        &mut *feed,
        &|ids| continuation(&l.tok, anchor, ids),
        l.model.cfg.ctx,
        &prompt,
        &o,
        &mut |s| {
            if let Err(e) = write!(out, "{s}") {
                werr.get_or_insert(e.to_string());
            }
        },
    )?;
    if let Some(e) = werr {
        return Err(e);
    }
    if o.stream {
        writeln!(out).map_err(|e| e.to_string())
    } else {
        say(out, format!("{shown}{text}"))
    }
}

/// `eval` for a model with its own tokenizer: loss, perplexity and bits per byte of `--data`.
/// `--quant f32|int8|int4` scores the model after a round trip through that precision.
pub fn eval(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let mut l = load(a.need("--model")?)?;
    let text = std::fs::read_to_string(a.need("--data")?).map_err(|e| format!("--data: {e}"))?;
    let q = super::gpt2::quant(a, "f32")?;
    if a.get("--quant").is_some() {
        let g = Gguf::parse(to_gguf(&l.model, q)).map_err(|e| e.to_string())?;
        l.model = from_gguf(&g).map_err(|e| e.to_string())?;
    }
    let max = l.model.cfg.ctx;
    let ctx = a.num("--ctx", 512usize.min(max))?;
    if ctx < 2 || ctx > max {
        return Err(format!("--ctx must be between 2 and {max} (got {ctx})"));
    }
    let mut ids: Vec<usize> = l.tok.prefix_bos().into_iter().collect();
    ids.extend(l.tok.encode(&text));
    let (loss, n) = mean_loss_ids(&l.model, &ids, ctx)?;
    let bpb = loss * n as f64 / std::f64::consts::LN_2 / text.len().max(1) as f64;
    say(
        out,
        format!(
            "tokens: {n} loss: {loss:.4} perplexity: {:.3} bits/byte: {bpb:.4}",
            loss.exp()
        ),
    )?;
    if let Some(p) = a.get("--json") {
        std::fs::write(
            p,
            format!(
                "{{\"tokens\":{n},\"bytes\":{},\"loss\":{loss},\"perplexity\":{},\"bits_per_byte\":{bpb}}}\n",
                text.len(),
                loss.exp()
            ),
        )
        .map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
