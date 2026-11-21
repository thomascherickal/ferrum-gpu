use super::*;

#[test]
fn add_bos_follows_the_file_then_llama_cpps_defaults() {
    let with = |model: &str, extra: &[(&str, Value)]| {
        let mut e = vec![("tokenizer.ggml.bos_token_id", Value::U32(0))];
        e.extend_from_slice(extra);
        build(model, &["<s>", "a"], &[], &e).unwrap().prefix_bos()
    };
    assert_eq!(with("gpt2", &[]), None); // Qwen-style: no BOS
    assert_eq!(
        with(
            "gpt2",
            &[("tokenizer.ggml.pre", Value::Str("llama-bpe".into()))]
        ),
        Some(0)
    );
    assert_eq!(
        with(
            "gpt2",
            &[("tokenizer.ggml.add_bos_token", Value::Bool(true))]
        ),
        Some(0)
    );
    assert_eq!(
        with(
            "llama",
            &[("tokenizer.ggml.add_bos_token", Value::Bool(false))]
        ),
        None
    );
    // no BOS id at all
    assert_eq!(build("llama", &["a"], &[], &[]).unwrap().prefix_bos(), None);
}

#[test]
fn missing_or_unsupported_tokenizers_are_named() {
    let bare = GgufBuilder::new().into_bytes();
    let e = GgufTokenizer::from_gguf(&Gguf::parse(bare).unwrap())
        .err()
        .unwrap();
    assert!(e.contains("no tokenizer.ggml.model"), "{e}");
    assert!(build("wpm", &["a"], &[], &[])
        .err()
        .unwrap()
        .contains("unsupported GGUF tokenizer model 'wpm'"));
    assert!(build("gpt2", &[], &[], &[])
        .err()
        .unwrap()
        .contains("empty"));
    let mut b = GgufBuilder::new();
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    let e = GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap())
        .err()
        .unwrap();
    assert!(e.contains("missing tokenizer.ggml.tokens"), "{e}");
}
