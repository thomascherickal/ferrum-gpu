use super::*;

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
