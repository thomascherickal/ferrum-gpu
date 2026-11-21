use super::*;

#[test]
fn hostile_metadata_is_refused_before_it_can_size_an_allocation() {
    let m = Llama::random(cfg(RopeType::Norm), 6, false, false);
    let u = |k: &'static str, v: u32| set(k, Value::U32(v));
    assert!(refused(
        &m,
        &set("general.architecture", Value::Str("gpt2".into())),
        &[]
    )
    .contains("only 'llama'"));
    assert!(refused(
        &m,
        &|kv| kv.retain(|(k, _)| k != "general.architecture"),
        &[]
    )
    .contains("missing"));
    assert!(refused(&m, &u("llama.block_count", 1_000_000), &[]).contains("exceeds"));
    assert!(refused(&m, &u("llama.attention.head_count", 0), &[]).contains("positive"));
    // no head_count and no key_length to fall back on
    let no_len = |kv: &mut Meta| {
        kv.retain(|(k, _)| !k.ends_with("attention.key_length"));
        set("llama.attention.head_count", Value::U32(0))(kv);
    };
    assert!(refused(&m, &no_len, &[]).contains("positive"));
    assert!(refused(&m, &u("llama.attention.head_count_kv", 3), &[]).contains("multiple"));
    assert!(refused(&m, &u("llama.rope.dimension_count", 5), &[]).contains("rope_dim"));
    assert!(refused(&m, &u("llama.rope.dimension_count", 16), &[]).contains("rope_dim"));
    assert!(refused(&m, &u("llama.embedding_length", 64), &[]).contains("embedding_length"));
    assert!(refused(&m, &u("llama.feed_forward_length", 7), &[]).contains("dims"));
    assert!(refused(&m, &u("llama.attention.key_length", 4), &[]).contains("rope_dim"));
    assert!(refused(&m, &set("llama.rope.freq_base", Value::U32(5)), &[]).contains("not an f32"));
    assert!(refused(
        &m,
        &set("llama.attention.layer_norm_rms_epsilon", Value::F32(0.0)),
        &[]
    )
    .contains("eps"));
    assert!(refused(&m, &set("llama.context_length", Value::U32(0)), &[]).contains("positive"));
    assert!(
        refused(&m, &set("llama.embedding_length", Value::I32(-1)), &[]).contains("non-negative")
    );
}

#[test]
fn defaults_apply_when_optional_keys_are_absent() {
    let mut c = cfg(RopeType::Norm);
    c.kv_heads = 4; // the default when head_count_kv is absent
    let m = Llama::random(c, 7, false, false);
    let strip = |kv: &mut Meta| {
        kv.retain(|(k, _)| {
            ![
                "head_count_kv",
                "key_length",
                "dimension_count",
                "freq_base",
                "layer_norm_rms_epsilon",
                "context_length",
            ]
            .iter()
            .any(|s| k.ends_with(s))
        })
    };
    let back = parse(write(&m, Quant::F32, &strip, &[])).unwrap();
    assert_eq!(back.cfg.kv_heads, 4);
    assert_eq!((back.cfg.head_dim, back.cfg.rope_dim), (8, 8));
    assert_eq!(
        (back.cfg.rope_base, back.cfg.eps, back.cfg.ctx),
        (10000.0, 1e-5, 2048)
    );
}
