//! Chat-prompt formatting for instruct GGUFs.
//!
//! Provenance: ported from ferrum-cpu `ferrum_core/src/chat_format.rs` (commit 56a9d3a); tests
//! changed to build tokenizers with our GGUF writer.
//!
//! No Jinja: the `tokenizer.chat_template` string is *fingerprinted* against
//! the four formats small llama/qwen models actually use, with a vocab-based
//! fallback when the key is absent. `render` returns text/special segments so
//! the runner can map each special token to its exact vocab id instead of
//! letting byte-BPE shred it. Single-turn only, by design.

use crate::gguf_tok::GgufTokenizer;

/// The single-turn chat templates Ferrum recognises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    /// Qwen2/Qwen3 (`<|im_start|>` …).
    ChatMl,
    /// Llama-3.x header tags.
    Llama3,
    /// `[INST] … [/INST]`.
    Llama2Inst,
    /// TinyLlama-chat style `<|user|>` / `<|assistant|>`.
    Zephyr,
}

/// One piece of a rendered prompt.
pub enum Segment {
    /// Ordinary text, encoded with the tokenizer.
    Text(String),
    /// A special token, mapped to its vocab id when the vocab has it.
    Special(&'static str),
}

/// Decide the chat format. `template` is `tokenizer.chat_template` if present;
/// `has_token` reports whether a piece exists verbatim in the vocab (the
/// instruct signal for template-less files — base models must NOT be wrapped,
/// so arch alone is never enough).
pub fn detect(
    template: Option<&str>,
    arch: Option<&str>,
    has_token: impl Fn(&str) -> bool,
) -> Option<ChatFormat> {
    detect_dyn(template, arch, &has_token)
}

fn detect_dyn(
    template: Option<&str>,
    arch: Option<&str>,
    has_token: &dyn Fn(&str) -> bool,
) -> Option<ChatFormat> {
    if let Some(t) = template {
        if t.contains("<|im_start|>") {
            return Some(ChatFormat::ChatMl);
        }
        if t.contains("<|start_header_id|>") {
            return Some(ChatFormat::Llama3);
        }
        if t.contains("[INST]") {
            return Some(ChatFormat::Llama2Inst);
        }
        if t.contains("<|user|>") {
            return Some(ChatFormat::Zephyr);
        }
    }
    match arch {
        Some("qwen2") if has_token("<|im_start|>") => Some(ChatFormat::ChatMl),
        Some("qwen3") if has_token("<|im_start|>") => Some(ChatFormat::ChatMl),
        Some("llama") if has_token("<|start_header_id|>") => Some(ChatFormat::Llama3),
        _ => None,
    }
}

/// Single-turn prompt in `fmt`, ending where the assistant should speak.
pub fn render(fmt: ChatFormat, user: &str) -> Vec<Segment> {
    use Segment::*;
    match fmt {
        ChatFormat::ChatMl => vec![
            Special("<|im_start|>"),
            Text(format!("user\n{user}")),
            Special("<|im_end|>"),
            Text("\n".into()),
            Special("<|im_start|>"),
            Text("assistant\n".into()),
        ],
        ChatFormat::Llama3 => vec![
            Special("<|start_header_id|>"),
            Text("user".into()),
            Special("<|end_header_id|>"),
            Text(format!("\n\n{user}")),
            Special("<|eot_id|>"),
            Special("<|start_header_id|>"),
            Text("assistant".into()),
            Special("<|end_header_id|>"),
            Text("\n\n".into()),
        ],
        ChatFormat::Llama2Inst => vec![Text(format!("[INST] {user} [/INST]"))],
        ChatFormat::Zephyr => vec![
            Special("<|user|>"),
            Text(format!("\n{user}")),
            Special("</s>"),
            Text("\n".into()),
            Special("<|assistant|>"),
            Text("\n".into()),
        ],
    }
}

/// The token that ends the assistant turn in `fmt`.
pub fn stop_token(fmt: ChatFormat) -> &'static str {
    match fmt {
        ChatFormat::ChatMl => "<|im_end|>",
        ChatFormat::Llama3 => "<|eot_id|>",
        ChatFormat::Llama2Inst | ChatFormat::Zephyr => "</s>",
    }
}

/// Encode segments: specials map to their atomic vocab id when present
/// (falling back to plain text encoding — degraded but functional).
pub fn encode_segments(tok: &GgufTokenizer, segs: &[Segment]) -> Vec<usize> {
    let mut ids = Vec::new();
    for s in segs {
        match s {
            Segment::Special(p) => match tok.token_id(p) {
                Some(id) => ids.push(id),
                None => ids.extend(tok.encode(p)),
            },
            Segment::Text(t) => ids.extend(tok.encode(t)),
        }
    }
    ids
}
#[cfg(test)]
mod tests;
