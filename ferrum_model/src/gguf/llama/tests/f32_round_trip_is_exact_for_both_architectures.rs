use super::*;

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
