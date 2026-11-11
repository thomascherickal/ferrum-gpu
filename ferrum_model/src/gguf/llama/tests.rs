use super::*;

fn cfg(ty: RopeType) -> Config {
    Config {
        vocab: 13,
        ctx: 12,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: ty,
        eps: 1e-5,
    }
}

fn parse(bytes: Vec<u8>) -> Result<Llama, GgufError> {
    from_gguf(&Gguf::parse(bytes)?)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn set(key: &'static str, v: Value) -> impl Fn(&mut Meta) {
    move |kv| {
        kv.retain(|(k, _)| k != key);
        kv.push((key.to_string(), v.clone()));
    }
}

fn refused(m: &Llama, edit: &dyn Fn(&mut Meta), skip: &[&str]) -> String {
    parse(write(m, Quant::F32, edit, skip)).unwrap_err().0
}

#[test]
fn f32_round_trip_is_exact_for_both_architectures() {
    for (ty, biases, tied) in [
        (RopeType::Norm, false, false),
        (RopeType::Norm, false, true),
        (RopeType::Neox, true, false),
        (RopeType::Neox, true, true),
    ] {
        let m = Llama::random(cfg(ty), 1, biases, tied);
        assert_eq!(
            parse(to_gguf(&m, Quant::F32)).unwrap(),
            m,
            "{ty:?} {biases} {tied}"
        );
    }
}

#[test]
fn the_architecture_decides_the_rope_pairing() {
    let norm = to_gguf(
        &Llama::random(cfg(RopeType::Norm), 1, false, false),
        Quant::F32,
    );
    let neox = to_gguf(
        &Llama::random(cfg(RopeType::Neox), 1, true, false),
        Quant::F32,
    );
    assert_eq!(
        Gguf::parse(norm)
            .unwrap()
            .string("general.architecture")
            .unwrap(),
        "llama"
    );
    assert_eq!(
        Gguf::parse(neox)
            .unwrap()
            .string("general.architecture")
            .unwrap(),
        "qwen2"
    );
}

#[test]
fn round_trip_gives_identical_logits_and_a_tied_file_has_no_output_tensor() {
    let m = Llama::random(cfg(RopeType::Norm), 2, false, true);
    let bytes = to_gguf(&m, Quant::F32);
    let g = Gguf::parse(bytes).unwrap();
    assert!(g.tensor("output.weight").is_none());
    let ids = [3, 1, 4, 1, 5];
    assert_eq!(
        from_gguf(&g).unwrap().forward_logits(&ids).unwrap(),
        m.forward_logits(&ids).unwrap()
    );
    let untied = Llama::random(cfg(RopeType::Norm), 2, false, false);
    assert!(Gguf::parse(to_gguf(&untied, Quant::F32))
        .unwrap()
        .tensor("output.weight")
        .is_some());
}

#[test]
fn quantised_round_trips_stay_close_and_q8_beats_q4() {
    let m = Llama::random(cfg(RopeType::Neox), 3, true, false);
    let ids = [2, 7, 1, 8];
    let want = m.forward_logits(&ids).unwrap();
    let err = |q| {
        let back = parse(to_gguf(&m, q)).unwrap();
        max_diff(&back.forward_logits(&ids).unwrap(), &want)
    };
    let (e8, e4) = (err(Quant::Q8_0), err(Quant::Q4_0));
    assert!(e8 < 0.3, "q8 err {e8}");
    assert!(e8 < e4, "q8 {e8} should beat q4 {e4}");
    // matrices whose input width is a multiple of 32 are really packed
    let g = Gguf::parse(to_gguf(&m, Quant::Q8_0)).unwrap();
    assert_eq!(
        g.tensor("blk.0.attn_q.weight").unwrap().ggml_type,
        GGML_Q8_0
    );
    assert_eq!(
        g.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
        GGML_F32
    );
    // a width that is not a multiple of 32 falls back to F32
    let mut c = cfg(RopeType::Norm);
    c.hidden = 48;
    let g = Gguf::parse(to_gguf(&Llama::random(c, 1, false, false), Quant::Q4_0)).unwrap();
    assert_eq!(
        g.tensor("blk.0.ffn_gate.weight").unwrap().ggml_type,
        GGML_Q4_0
    );
    assert_eq!(
        g.tensor("blk.0.ffn_down.weight").unwrap().ggml_type,
        GGML_F32
    );
    c.dim = 8;
    c.head_dim = 2;
    c.rope_dim = 2;
    let g = Gguf::parse(to_gguf(&Llama::random(c, 1, false, false), Quant::Q4_0)).unwrap();
    assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ggml_type, GGML_F32);
}

#[test]
fn rope_scaling_round_trips_through_metadata_and_the_freq_tensor() {
    let mut m = Llama::random(cfg(RopeType::Norm), 4, false, false);
    m.rope = RopeScaling {
        freq_factors: vec![1.0, 2.0, 4.0, 8.0],
        linear_factor: 2.0,
    };
    let back = parse(to_gguf(&m, Quant::F32)).unwrap();
    assert_eq!(back.rope, m.rope);
    assert_eq!(back, m);
}

#[test]
fn scaling_policy_follows_llama_cpp() {
    let m = Llama::random(cfg(RopeType::Norm), 5, false, false);
    let t = |v: &str| set("llama.rope.scaling.type", Value::Str(v.into()));
    assert!(refused(&m, &t("yarn"), &[]).contains("'yarn' is not supported"));
    assert!(refused(&m, &t("longrope"), &[]).contains("longrope"));
    let attn = set("llama.rope.scaling.attn_factor", Value::F32(1.5));
    assert!(refused(&m, &attn, &[]).contains("attn_factor"));
    let none_with_factor = |kv: &mut Meta| {
        t("none")(kv);
        set("llama.rope.scaling.factor", Value::F32(2.0))(kv);
    };
    assert!(refused(&m, &none_with_factor, &[]).contains("\"none\""));
    // "none" with factor 1, a missing type with a factor (linear) and a zero factor are fine
    let ok = |edit: &dyn Fn(&mut Meta)| parse(write(&m, Quant::F32, edit, &[])).unwrap();
    assert_eq!(ok(&t("none")).rope.linear_factor, 1.0);
    assert_eq!(
        ok(&set("llama.rope.scaling.factor", Value::F32(0.0)))
            .rope
            .linear_factor,
        1.0
    );
    assert_eq!(
        ok(&set("llama.rope.scale_linear", Value::F32(4.0)))
            .rope
            .linear_factor,
        4.0
    );
    // a wrong-length rope_freqs.weight is refused
    let mut b = GgufBuilder::new();
    let good = to_gguf(&m, Quant::F32);
    let g = Gguf::parse(good).unwrap();
    g.kv.iter().for_each(|(k, v)| b.meta(k, v.clone()));
    for t in &g.tensors {
        b.tensor_f32(&t.name, &t.dims, &g.tensor_f32(&t.name).unwrap());
    }
    b.tensor_f32("rope_freqs.weight", &[3], &[1.0; 3]);
    assert!(parse(b.into_bytes())
        .unwrap_err()
        .0
        .contains("rope_freqs.weight"));
}

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
