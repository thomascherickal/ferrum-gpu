use super::*;

fn flat(segs: &[Segment]) -> String {
    segs.iter()
        .map(|s| match s {
            Segment::Text(t) => t.as_str(),
            Segment::Special(p) => p,
        })
        .collect()
}

#[test]
fn detect_fingerprints_template_string() {
    let t = |s: &str| detect(Some(s), None, |_| false);
    assert_eq!(t("{% if %}<|im_start|>…"), Some(ChatFormat::ChatMl));
    assert_eq!(t("…<|start_header_id|>…"), Some(ChatFormat::Llama3));
    assert_eq!(t("…[INST]…"), Some(ChatFormat::Llama2Inst));
    assert_eq!(t("…<|user|>…"), Some(ChatFormat::Zephyr));
    assert_eq!(t("unrecognized"), None);
}

#[test]
fn detect_falls_back_by_arch_only_with_vocab_evidence() {
    assert_eq!(
        detect(None, Some("qwen2"), |p| p == "<|im_start|>"),
        Some(ChatFormat::ChatMl)
    );
    assert_eq!(
        detect(None, Some("qwen3"), |p| p == "<|im_start|>"),
        Some(ChatFormat::ChatMl)
    );
    assert_eq!(detect(None, Some("qwen2"), |_| false), None); // base model: never wrap
    assert_eq!(detect(None, Some("qwen3"), |_| false), None);
    assert_eq!(
        detect(None, Some("llama"), |p| p == "<|start_header_id|>"),
        Some(ChatFormat::Llama3)
    );
    assert_eq!(detect(None, Some("llama"), |_| false), None);
    assert_eq!(detect(None, None, |_| true), None);
}

#[test]
fn render_golden_strings() {
    assert_eq!(
        flat(&render(ChatFormat::ChatMl, "hi")),
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        flat(&render(ChatFormat::Llama3, "hi")),
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert_eq!(
        flat(&render(ChatFormat::Llama2Inst, "hi")),
        "[INST] hi [/INST]"
    );
    assert_eq!(
        flat(&render(ChatFormat::Zephyr, "hi")),
        "<|user|>\nhi</s>\n<|assistant|>\n"
    );
}

#[test]
fn stop_tokens_per_format() {
    assert_eq!(stop_token(ChatFormat::ChatMl), "<|im_end|>");
    assert_eq!(stop_token(ChatFormat::Llama3), "<|eot_id|>");
    assert_eq!(stop_token(ChatFormat::Llama2Inst), "</s>");
    assert_eq!(stop_token(ChatFormat::Zephyr), "</s>");
}

fn tok() -> GgufTokenizer {
    use ferrum_gguf::{GgufBuilder, Value};
    let mut b = GgufBuilder::new();
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    let toks = ["a", "b", "<|im_start|>", "<|im_end|>"];
    b.meta(
        "tokenizer.ggml.tokens",
        Value::Arr(toks.iter().map(|s| Value::Str((*s).into())).collect()),
    );
    GgufTokenizer::from_gguf(&ferrum_gguf::Gguf::parse(b.into_bytes()).unwrap()).unwrap()
}

#[test]
fn segments_map_specials_to_their_ids_and_text_to_pieces() {
    let t = tok();
    let ids = encode_segments(
        &t,
        &[
            Segment::Special("<|im_start|>"),
            Segment::Text("ab".into()),
            Segment::Special("<|im_end|>"),
        ],
    );
    assert_eq!(ids, vec![2, 0, 1, 3]);
    // a special the vocab lacks degrades to plain text (here: unknown bytes encode to nothing)
    let ids = encode_segments(&t, &[Segment::Special("a")]);
    assert_eq!(ids, vec![0]); // "a" is a token: used as it is
    let ids = encode_segments(&t, &[Segment::Special("ab")]);
    assert_eq!(ids, vec![0, 1]); // "ab" is not: encoded as the text "ab"
}
