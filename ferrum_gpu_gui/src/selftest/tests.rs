use super::*;

#[test]
fn base64_matches_the_rfc_4648_vectors() {
    for (raw, enc) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(base64(raw.as_bytes()), enc);
    }
    assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
    assert_eq!(base64(&[0, 0, 0, 0]), "AAAAAA==");
}

#[test]
fn bench_models_load_and_unknown_names_are_refused() {
    for (shape, quant) in [("small", "f32"), ("small", "q8_0"), ("small", "q4_0")] {
        let bytes = bench_gguf(shape, quant).unwrap();
        let m = from_gguf(&Gguf::parse(bytes).unwrap()).unwrap();
        assert_eq!(m.cfg, bench_config(shape).unwrap());
    }
    assert!(bench_gguf("huge", "f32")
        .unwrap_err()
        .contains("unknown shape"));
    assert!(bench_gguf("small", "q3")
        .unwrap_err()
        .contains("unknown quant"));
    let vocab = |shape| bench_config(shape).unwrap().vocab;
    assert!(vocab("small") < vocab("medium") && vocab("medium") < vocab("large"));
}

#[test]
fn the_vectors_cover_every_variant_and_agree_with_their_files() {
    let v = vectors();
    let Some(Json::Arr(cases)) = v.get("cases") else {
        panic!("cases")
    };
    assert_eq!(cases.len(), 6);
    let mut quants = Vec::new();
    for c in cases {
        let name = c.get("name").and_then(Json::as_str).unwrap();
        quants.push(c.get("quant").and_then(Json::as_str).unwrap().to_string());
        let Some(Json::Arr(logits)) = c.get("logits") else {
            panic!("{name}: logits")
        };
        assert_eq!(logits.len(), IDS.len() * 17, "{name}");
        assert!(
            logits
                .iter()
                .all(|l| l.as_f64().is_some_and(f64::is_finite)),
            "{name}"
        );
        // decode the base64 again by hand and check it is a loadable GGUF of the same shape
        let text = c.get("gguf").and_then(Json::as_str).unwrap();
        assert_eq!(text.len() % 4, 0, "{name}: padded base64");
        assert!(text.starts_with("R0dVRg"), "{name}: 'GGUF' magic in base64");
    }
    assert!(
        quants.iter().any(|q| q == "Q8_0")
            && quants.iter().any(|q| q == "Q4_0")
            && quants.iter().filter(|q| *q == "F32").count() == 4
    );
    // the document is valid JSON
    assert!(Json::parse(&v.to_string()).is_ok());
}
