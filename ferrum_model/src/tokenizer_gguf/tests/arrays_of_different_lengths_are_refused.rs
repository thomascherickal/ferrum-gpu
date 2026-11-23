use super::*;

#[test]
fn arrays_of_different_lengths_are_refused() {
    let t = small();
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(TOKENIZER_ARCH.into()));
    b.meta("tokenizer.ggml.tokens", strings(t.gguf_tokens()));
    b.meta("tokenizer.ggml.merges", strings(t.gguf_merges()));
    b.meta("tokenizer.ggml.token_type", Value::Arr(vec![Value::I32(1)]));
    let e = from_gguf_bytes(&b.into_bytes()).unwrap_err();
    assert!(e.starts_with("gguf vocab"), "{e}");
}
