use super::*;

#[test]
fn a_file_without_the_embedding_is_refused() {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    for k in ["context_length", "embedding_length"] {
        b.meta(&format!("gpt2.{k}"), Value::U32(8));
    }
    b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert_eq!(e.to_string(), "gguf: missing tensor 'token_embd.weight'");
}
