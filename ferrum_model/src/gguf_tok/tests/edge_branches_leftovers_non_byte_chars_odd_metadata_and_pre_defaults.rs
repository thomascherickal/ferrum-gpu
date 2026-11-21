use super::*;

#[test]
fn edge_branches_leftovers_non_byte_chars_odd_metadata_and_pre_defaults() {
    let t = bpe();
    // "lo" has no merge and no token: it falls back to its byte tokens
    assert_eq!(t.encode("lo"), vec![2, 3]);
    // a char outside the byte alphabet decodes to its own UTF-8
    let u = build("gpt2", &["a", "\u{20ac}"], &[], &[]).unwrap();
    assert_eq!(u.decode(&[0, 1]), "a\u{20ac}");
    // a non-string entry in the token array becomes an empty token
    let v = build("gpt2", &["a"], &[], &[]).unwrap();
    assert_eq!(v.vocab_size(), 1);
    let mut b = GgufBuilder::new();
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    b.meta("tokenizer.ggml.tokens", Value::Arr(vec![Value::U32(1)]));
    let w = GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap();
    assert_eq!(w.decode(&[0]), "");
    // SPM decode skips ids out of range
    let s = build("llama", &["a"], &[], &[]).unwrap();
    assert_eq!(s.decode(&[0, 99]), "a");
    // every pre-tokenizer default that implies a BOS
    for pre in [
        "llama3",
        "llama-v3",
        "llama-bpe",
        "falcon3",
        "falcon-h1",
        "pixtral",
        "midm-2.0",
        "lfm2",
        "jina-v5-nano",
        "tekken",
        "chameleon",
    ] {
        let g = build(
            "gpt2",
            &["<s>", "a"],
            &[],
            &[
                ("tokenizer.ggml.bos_token_id", Value::U32(0)),
                ("tokenizer.ggml.pre", Value::Str(pre.into())),
            ],
        )
        .unwrap();
        assert_eq!(g.prefix_bos(), Some(0), "{pre}");
    }
}
