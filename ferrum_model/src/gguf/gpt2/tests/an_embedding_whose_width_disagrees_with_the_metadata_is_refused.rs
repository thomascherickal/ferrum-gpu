use super::*;

/// A GPT-2 file header: width 8, context 8, one layer, feed-forward 16 and `heads` heads.
fn header(heads: u32) -> GgufBuilder {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    for (k, v) in [
        ("context_length", 8u32),
        ("embedding_length", 8),
        ("block_count", 1),
        ("feed_forward_length", 16),
        ("attention.head_count", heads),
    ] {
        b.meta(&format!("gpt2.{k}"), Value::U32(v));
    }
    b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    b
}

#[test]
fn an_embedding_whose_width_disagrees_with_the_metadata_is_refused() {
    let m = Gpt2::random(cfg(8), 8);
    let mut b = header(2);
    // Declared width 8, but the tensor's contiguous dimension is 4.
    b.tensor_f32("token_embd.weight", &[4, 26], &m.tok);
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert!(
        e.to_string().contains("must be [embedding_length, vocab]"),
        "{e}"
    );
}

#[test]
fn a_tensor_of_the_wrong_size_is_named() {
    let m = Gpt2::random(cfg(8), 7);
    // Same file but claim heads that do not divide the width.
    let mut b = header(3);
    b.tensor_f32("token_embd.weight", &[8, 13], &m.tok);
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert!(e.to_string().contains("divisible"), "{e}");
    // Right metadata, truncated position table.
    let mut b = header(2);
    b.tensor_f32("token_embd.weight", &[8, 13], &m.tok);
    b.tensor_f32("position_embd.weight", &[8, 2], &m.pos[..16]);
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert!(
        e.to_string().contains("position_embd.weight") && e.to_string().contains("expected 64"),
        "{e}"
    );
}
