use super::*;
use crate::cmds::run as cli;
use ferrum_gguf::Value;
use ferrum_gpu_gui::engine::preset;
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

#[test]
fn a_chat_model_is_wrapped_in_its_template_and_prints_only_the_reply() {
    let (m, _) = model("chat", &TOKS, true);
    let p = m.display();
    let chat = ok(&format!(
        "generate --model {p} --prompt ab --max-new 4 --device cpu"
    ));
    assert!(
        !chat.contains("<|im_start|>") && !chat.starts_with("ab"),
        "{chat:?}"
    );
    // raw: no template, the prompt is echoed
    let raw = ok(&format!(
        "generate --model {p} --prompt abab --max-new 4 --device cpu --raw"
    ));
    assert!(raw.starts_with("abab"), "{raw:?}");
    // the same seed gives the same text
    assert_eq!(
        chat,
        ok(&format!(
            "generate --model {p} --prompt ab --max-new 4 --device cpu"
        ))
    );
    // streaming prints the same characters, prompt first in raw mode
    let s = ok(&format!(
        "generate --model {p} --prompt abab --max-new 4 --device cpu --raw --stream"
    ));
    assert_eq!(s, raw);
    // run-gguf adds the device and shape lines
    let v = ok(&format!(
        "run-gguf --model {p} --prompt ab --max-new 2 --device cpu"
    ));
    assert!(
        v.contains("device: CPU") && v.contains("model: vocab 12"),
        "{v}"
    );
}

#[test]
fn ids_bypass_the_template_and_bad_ids_are_named() {
    let (m, _) = model("ids", &TOKS, true);
    let p = m.display();
    let a = [
        "generate",
        "--model",
        &p.to_string(),
        "--ids",
        "0 1",
        "--max-new",
        "3",
        "--device",
        "cpu",
    ]
    .map(String::from);
    let mut out = Vec::new();
    cli(&a, &mut out).unwrap();
    assert!(!out.is_empty());
    let bad = [
        "generate",
        "--model",
        &p.to_string(),
        "--ids",
        "0 x",
        "--device",
        "cpu",
    ]
    .map(String::from);
    assert!(cli(&bad, &mut Vec::new())
        .unwrap_err()
        .contains("'x' is not an id"));
    let empty = [
        "generate",
        "--model",
        &p.to_string(),
        "--ids",
        "",
        "--device",
        "cpu",
    ]
    .map(String::from);
    assert!(cli(&empty, &mut Vec::new())
        .unwrap_err()
        .contains("prompt is empty"));
    assert!(err(&format!("generate --model {p} --device cpu")).contains("--prompt"));
}

#[test]
fn eval_scores_with_the_tokenizer_and_validates_ctx_and_quant() {
    let (m, data) = model("eval", &TOKS, false);
    let (p, d) = (m.display(), data.display());
    let s = ok(&format!("eval --model {p} --data {d}"));
    assert!(s.contains("perplexity:") && s.contains("bits/byte:"), "{s}");
    // ferrum-cpu's name for the same command gives the same report
    let alias = ok(&format!("eval-gguf --model {p} --data {d}"));
    assert_eq!(alias, s);
    let json = m.with_file_name("e.json");
    ok(&format!(
        "eval --model {p} --data {d} --ctx 4 --quant int8 --json {}",
        json.display()
    ));
    let j = std::fs::read_to_string(&json).unwrap();
    assert!(
        j.contains("\"bits_per_byte\"") && j.contains("\"bytes\":14"),
        "{j}"
    );
    assert!(err(&format!("eval --model {p} --data {d} --ctx 1"))
        .contains("--ctx must be between 2 and"));
    assert!(err(&format!("eval --model {p} --data {d} --quant fp9")).contains("fp9"));
    assert!(err(&format!("eval --model {p} --data /nonexistent-ferrum")).contains("--data"));
}

#[test]
fn a_tokenizer_larger_than_the_model_is_refused() {
    let many: Vec<String> = (0..20).map(|i| format!("t{i}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    let (m, _) = model("big", &refs, false);
    let e = load(&m.display().to_string()).err().unwrap();
    assert!(e.contains("20 tokens but the model only 12"), "{e}");
    assert!(has_tokenizer(&m.display().to_string()));
    assert!(!has_tokenizer("/nonexistent-ferrum"));
}

#[test]
fn only_instruct_files_get_a_chat_format() {
    let (with, _) = model("tpl", &TOKS, true);
    assert_eq!(
        load(&with.display().to_string()).unwrap().chat,
        Some(ChatFormat::ChatMl)
    );
    let (without, _) = model("notpl", &TOKS, false);
    assert!(load(&without.display().to_string()).unwrap().chat.is_none());
}

#[test]
fn a_continuation_keeps_its_leading_space() {
    use ferrum_gguf::GgufBuilder;
    let mut b = GgufBuilder::new();
    b.meta("tokenizer.ggml.model", Value::Str("llama".into()));
    b.meta(
        "tokenizer.ggml.tokens",
        arr(&["<unk>", "\u{2581}a", "b", "\u{2581}c"]),
    );
    let t = GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap();
    assert_eq!(continuation(&t, 1, &[2]), "b");
    assert_eq!(continuation(&t, 1, &[3]), " c");
    assert_eq!(continuation(&t, 1, &[]), "");
}

#[test]
fn a_stop_id_ends_the_reply_without_being_printed() {
    let tok = {
        use ferrum_gguf::GgufBuilder;
        let mut b = GgufBuilder::new();
        b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
        b.meta("tokenizer.ggml.tokens", arr(&["a", "b", "c", "d"]));
        GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap()
    };
    // The decoder says "a", "a", then "d" (the stop id) forever.
    let mut calls = 0;
    let mut feed = |_t: usize| -> Result<Vec<f32>, String> {
        calls += 1;
        let mut l = vec![0.0f32; 4];
        l[if calls < 3 { 0 } else { 3 }] = 9.0;
        Ok(l)
    };
    let o = crate::cmds::gpt2::gen::Opts {
        max_new: 10,
        params: Params::default(),
        seed: 1,
        stops: vec![],
        stream: false,
        stop_ids: vec![3],
    };
    let (text, n) = run(&mut feed, &|ids| tok.decode(ids), 64, &[1], &o, &mut |_| {}).unwrap();
    assert_eq!((text.as_str(), n), ("aa", 2));
}
