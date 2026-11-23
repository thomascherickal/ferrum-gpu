use super::*;
use crate::commands::run as cli;
use crate::engine::preset;
use ferrum_gguf::Value;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::to_gguf_with_meta;
use ferrum_model::sampling::Params;
use std::path::PathBuf;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn ok(s: &str) -> String {
    let mut out = Vec::new();
    cli(&words(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

fn err(s: &str) -> String {
    cli(&words(s), &mut Vec::new()).unwrap_err()
}

fn arr(items: &[&str]) -> Value {
    Value::Arr(items.iter().map(|s| Value::Str((*s).into())).collect())
}

/// A tiny Llama with a 12-token ChatML-style vocabulary, written to a temp file.
fn model(tag: &str, tokens: &[&str], template: bool) -> (PathBuf, PathBuf) {
    let d = std::env::temp_dir().join(format!("ferrum_llm_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let mut cfg = preset("tiny").unwrap();
    cfg.vocab = 12;
    let m = Llama::random(cfg, 3, false, false);
    let mut meta = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::Str("gpt2".into()),
        ),
        ("tokenizer.ggml.tokens".to_string(), arr(tokens)),
        ("tokenizer.ggml.eos_token_id".to_string(), Value::U32(7)),
    ];
    if template {
        meta.push((
            "tokenizer.chat_template".into(),
            Value::Str("{{ '<|im_start|>' }}".into()),
        ));
    }
    let p = d.join("m.gguf");
    std::fs::write(&p, to_gguf_with_meta(&m, Quant::F32, &meta)).unwrap();
    let data = d.join("t.txt");
    std::fs::write(&data, "abab abab abab").unwrap();
    (p, data)
}

const TOKS: [&str; 12] = [
    "a",
    "b",
    "c",
    "d",
    "\u{120}",
    "ab",
    "<|im_start|>",
    "<|im_end|>",
    "x",
    "y",
    "z",
    "w",
];

mod a_chat_model_is_wrapped_in_its_template_and_prints_only_the_reply;
mod a_stop_id_ends_the_reply_without_being_printed;
mod eval_scores_with_the_tokenizer_and_validates_ctx_and_quant;
mod models_that_cannot_be_loaded_or_reported_are_named;
