use super::*;

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
