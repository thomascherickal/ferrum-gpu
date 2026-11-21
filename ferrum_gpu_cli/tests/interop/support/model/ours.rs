use super::*;

/// Our greedy continuation of [`PROMPT`] with `model` (the prompt, then the text).
pub fn ours(model: &Path) -> String {
    ferrum_ok(&[
        "generate",
        "--device",
        "cpu",
        "--model",
        s(model),
        "--prompt",
        PROMPT,
        "--max-new",
        &NEW_TOKENS.to_string(),
    ])
    .trim_end()
    .to_string()
}
