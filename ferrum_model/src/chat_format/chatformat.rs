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

pub(super) fn detect_dyn(
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
