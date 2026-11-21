use super::*;

/// Reference values for the Node-level tests of the JS reader: a tensor decoded by the Rust
/// reader, the device repack of a quantised one, and the RoPE tables the shaders will read.
pub(super) fn probe(bytes: &[u8], model: &Llama, quant: Quant) -> Json {
    let g = Gguf::parse(bytes.to_vec()).expect("own file parses");
    let name = "blk.0.ffn_down.weight"; // [hidden, dim]: input width 64, a whole number of blocks
    let decoded = g.tensor_f32(name).expect("decodes");
    let mut p = vec![
        ("tensor", Json::Str(name.into())),
        (
            "decoded",
            Json::Arr(
                decoded
                    .iter()
                    .take(96)
                    .map(|&v| Json::Num(f64::from(v)))
                    .collect(),
            ),
        ),
        ("elements", Json::Num(decoded.len() as f64)),
    ];
    let kind = match quant {
        Quant::Q8_0 => Some(QKind::Q8_0),
        Quant::Q4_0 => Some(QKind::Q4_0),
        // ponytail: the browser self-test builds only f32, Q8_0 and Q4_0 cases; give other precisions
        // an arm here when a case for them is added
        _ => None,
    };
    if let Some(kind) = kind {
        let (k, n) = (model.cfg.hidden, model.cfg.dim);
        let words = pack(kind, g.tensor_bytes(name).expect("raw"), k, n).expect("packs");
        p.push((
            "packed",
            Json::Arr(
                words
                    .iter()
                    .take(120)
                    .map(|&w| Json::Num(f64::from(w)))
                    .collect(),
            ),
        ));
        p.push(("packed_len", Json::Num(words.len() as f64)));
    }
    let (cos, sin) = rope_tables(
        0,
        model.cfg.ctx,
        model.cfg.rope_dim,
        model.cfg.rope_base,
        &model.rope,
    );
    p.push((
        "rope_cos",
        Json::Arr(cos.iter().map(|&v| Json::Num(f64::from(v))).collect()),
    ));
    p.push((
        "rope_sin",
        Json::Arr(sin.iter().map(|&v| Json::Num(f64::from(v))).collect()),
    ));
    Json::obj_vec(p)
}

/// All cases as one JSON document: `{cases: [{name, quant, tolerance, ids, vocab, gguf, logits}]}`.
pub fn vectors() -> Json {
    let cases = cases()
        .into_iter()
        .map(|(name, model, quant)| {
            let bytes = to_gguf(&model, quant);
            // The oracle sees exactly what the browser will load.
            let loaded = from_gguf(&Gguf::parse(bytes.clone()).expect("own file parses"))
                .expect("own file loads");
            let logits = loaded.forward_logits(&IDS).expect("oracle forward");
            Json::obj([
                ("name", Json::Str(name.into())),
                ("quant", Json::Str(format!("{quant:?}"))),
                ("tolerance", Json::Num(1e-4)),
                (
                    "ids",
                    Json::Arr(IDS.iter().map(|&i| Json::Num(i as f64)).collect()),
                ),
                ("vocab", Json::Num(model.cfg.vocab as f64)),
                ("gguf", Json::Str(base64(&bytes))),
                ("probe", probe(&bytes, &loaded, quant)),
                (
                    "logits",
                    Json::Arr(logits.iter().map(|&l| Json::Num(f64::from(l))).collect()),
                ),
            ])
        })
        .collect();
    Json::obj([("cases", Json::Arr(cases))])
}
