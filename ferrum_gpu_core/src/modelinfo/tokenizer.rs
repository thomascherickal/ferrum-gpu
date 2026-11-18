//! What a model file says about its tokenizer, from the header alone: the family, the vocabulary
//! size, and the chat format an instruct model is prompted with (the same recognition `generate`
//! uses, so the Models tab and the commands agree).

use ferrum_gguf::{Gguf, Value};
use ferrum_model::chat_format::{detect, ChatFormat};

/// The tokenizer of a model file.
#[derive(Clone, Debug, PartialEq)]
pub struct TokInfo {
    /// `tokenizer.ggml.model` (`gpt2` for byte-level BPE, `llama` for SentencePiece).
    pub model: String,
    /// Entries in the vocabulary.
    pub tokens: usize,
    /// The chat format of an instruct model, `None` for a base model.
    pub chat_format: Option<&'static str>,
}

/// The name shown for a chat format.
pub fn format_name(f: ChatFormat) -> &'static str {
    match f {
        ChatFormat::ChatMl => "ChatML",
        ChatFormat::Llama3 => "Llama-3",
        ChatFormat::Llama2Inst => "[INST]",
        ChatFormat::Zephyr => "Zephyr",
    }
}

/// The tokenizer facts of `g`, `None` when the file has no vocabulary.
pub fn tokenizer_info(g: &Gguf) -> Option<TokInfo> {
    let Some(Value::Arr(tokens)) = g.get("tokenizer.ggml.tokens") else {
        return None;
    };
    let has = |p: &str| tokens.iter().any(|t| matches!(t, Value::Str(s) if s == p));
    let chat_format = detect(
        g.string("tokenizer.chat_template").ok(),
        g.string("general.architecture").ok(),
        has,
    )
    .map(format_name);
    Some(TokInfo {
        model: g
            .string("tokenizer.ggml.model")
            .unwrap_or("unknown")
            .to_string(),
        tokens: tokens.len(),
        chat_format,
    })
}

#[cfg(test)]
mod tests;
