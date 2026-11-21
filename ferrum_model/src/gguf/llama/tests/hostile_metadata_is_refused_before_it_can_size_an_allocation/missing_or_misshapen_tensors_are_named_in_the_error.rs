use super::*;

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
