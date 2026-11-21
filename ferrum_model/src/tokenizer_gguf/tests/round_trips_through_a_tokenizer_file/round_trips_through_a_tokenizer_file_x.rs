use super::*;

#[test]
fn round_trips_through_a_tokenizer_file() {
    let t = small();
    let back = from_gguf_bytes(&to_gguf_bytes(&t)).unwrap();
    assert_eq!(back.encode_state(), t.encode_state());
    assert_eq!(back.encode("abab ünï"), t.encode("abab ünï"));
}

#[test]
fn reads_a_ferrum_cpu_tokenizer_file() {
    let t = from_gguf_bytes(TOK).unwrap();
    assert_eq!(t.vocab_size(), 320);
    let text = "Der Hund läuft 日本語 🌸 it's";
    assert_eq!(t.decode(&t.encode(text)), text);
    // writing it back gives a file that reads back to the same tokenizer
    assert_eq!(
        from_gguf_bytes(&to_gguf_bytes(&t)).unwrap().encode_state(),
        t.encode_state()
    );
}

#[test]
fn a_model_file_from_ferrum_cpu_carries_the_same_tokenizer() {
    let g = Gguf::parse(MODEL.to_vec()).unwrap();
    let from_model = from_gguf(&g).unwrap();
    let from_file = from_gguf_bytes(TOK).unwrap();
    assert_eq!(from_model.encode_state(), from_file.encode_state());
}

#[test]
fn rebuilds_from_arrays_when_the_state_key_is_absent() {
    let t = small();
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(TOKENIZER_ARCH.into()));
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    b.meta("tokenizer.ggml.tokens", strings(t.gguf_tokens()));
    b.meta("tokenizer.ggml.merges", strings(t.gguf_merges()));
    b.meta(
        "tokenizer.ggml.token_type",
        Value::Arr(t.gguf_token_types().into_iter().map(Value::I32).collect()),
    );
    let back = from_gguf_bytes(&b.into_bytes()).unwrap();
    assert_eq!(back.encode_state(), t.encode_state());
}

#[test]
fn rejects_bad_files() {
    assert!(from_gguf_bytes(b"").unwrap_err().contains("truncated"));
    assert!(from_gguf_bytes(b"not gguf at all")
        .unwrap_err()
        .contains("not a GGUF"));
    assert!(from_gguf_bytes(&MODEL[..40]).is_err());
    // a model file is not a tokenizer file
    assert!(from_gguf_bytes(MODEL)
        .unwrap_err()
        .contains("not a ferrum-tokenizer"));
}
