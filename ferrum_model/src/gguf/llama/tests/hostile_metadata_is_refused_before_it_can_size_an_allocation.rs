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

#[test]
fn missing_or_misshapen_tensors_are_named_in_the_error() {
    let m = Llama::random(cfg(RopeType::Norm), 8, false, false);
    for name in [
        "token_embd.weight",
        "blk.1.attn_q.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
    ] {
        assert!(refused(&m, &|_| {}, &[name]).contains(name), "{name}");
    }
    // the output head, when present, must be [dim, vocab]
    let mut wrong = m.clone();
    wrong.cfg.vocab = 11;
    wrong.tok.truncate(11 * 32);
    wrong.head.truncate(32 * 11);
    let mut b = GgufBuilder::new();
    let g = Gguf::parse(to_gguf(&m, Quant::F32)).unwrap();
    g.kv.iter().for_each(|(k, v)| b.meta(k, v.clone()));
    for t in &g.tensors {
        let data = g.tensor_f32(&t.name).unwrap();
        if t.name == "token_embd.weight" {
            b.tensor_f32(&t.name, &[32, 11], &wrong.tok);
        } else {
            b.tensor_f32(&t.name, &t.dims, &data);
        }
    }
    assert!(parse(b.into_bytes())
        .unwrap_err()
        .0
        .contains("output.weight"));
    // a 3-D embedding is refused
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("llama".into()));
    b.tensor_f32("token_embd.weight", &[32, 1, 13], &vec![0.0; 32 * 13]);
    assert!(parse(b.into_bytes()).is_err());
}

#[test]
fn biases_outside_qkv_are_refused_not_dropped() {
    let m = Llama::random(cfg(RopeType::Norm), 9, false, false);
    for extra in [
        "blk.0.attn_output.bias",
        "blk.1.ffn_gate.bias",
        "blk.0.ffn_up.bias",
        "blk.0.ffn_down.bias",
        "output.bias",
    ] {
        let mut b = GgufBuilder::new();
        let g = Gguf::parse(to_gguf(&m, Quant::F32)).unwrap();
        g.kv.iter().for_each(|(k, v)| b.meta(k, v.clone()));
        for t in &g.tensors {
            b.tensor_f32(&t.name, &t.dims, &g.tensor_f32(&t.name).unwrap());
        }
        b.tensor_f32(extra, &[32], &[0.0; 32]);
        let e = parse(b.into_bytes()).unwrap_err().0;
        assert!(
            e.contains(extra) || e.contains("output.bias"),
            "{extra}: {e}"
        );
    }
}

#[test]
fn a_wrongly_sized_qkv_bias_is_refused() {
    let m = Llama::random(cfg(RopeType::Neox), 10, true, false);
    let mut b = GgufBuilder::new();
    let g = Gguf::parse(to_gguf(&m, Quant::F32)).unwrap();
    g.kv.iter().for_each(|(k, v)| b.meta(k, v.clone()));
    for t in &g.tensors {
        if t.name == "blk.0.attn_k.bias" {
            b.tensor_f32(&t.name, &[3], &[0.0; 3]);
        } else {
            b.tensor_f32(&t.name, &t.dims, &g.tensor_f32(&t.name).unwrap());
        }
    }
    assert!(parse(b.into_bytes()).unwrap_err().0.contains("attn_k.bias"));
}
