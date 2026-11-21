use super::*;

#[test]
fn models_move_through_gguf_in_every_precision() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    assert_eq!(
        get(p, "/api/model.gguf").status,
        404,
        "nothing to export yet"
    );
    post(p, "/api/model", r#"{"preset":"small","seed":3}"#);
    let gen = |port: u16| post(port, "/api/generate", r#"{"prompt":"hello ","max_new":10}"#).json();
    let want = gen(p);
    let f32_bytes = get(p, "/api/model.gguf").body;
    assert!(f32_bytes.starts_with(b"GGUF"));
    let (q8, q4) = (
        get(p, "/api/model.gguf?quant=q8_0"),
        get(p, "/api/model.gguf?quant=q4_0"),
    );
    assert!(
        q8.head.contains("attachment")
            && q8.body.len() < f32_bytes.len() / 2
            && q4.body.len() < q8.body.len()
    );
    // the exact export loads into a different server and decodes identically
    let (srv2, _a2) = serve(cpu_app());
    let p2 = srv2.addr.port();
    let up = call(p2, "PUT", "/api/model.gguf", &f32_bytes);
    assert_eq!(up.status, 200, "{}", up.text());
    assert_eq!(field(&up.json(), "preset").as_str(), Some("small"));
    assert_eq!(gen(p2), want, "an f32 round trip is exact");
    // quantised exports load too (a different model, so only shape and sanity are checked)
    for q in [&q8, &q4] {
        let up = call(p2, "PUT", "/api/model.gguf", &q.body);
        assert_eq!(up.status, 200, "{}", up.text());
        assert_eq!(
            field(&gen(p2), "tokens").to_string().matches(',').count(),
            9
        );
    }
    // garbage and truncated files are refused with a reason
    for bad in [&b"not a gguf"[..], &f32_bytes[..f32_bytes.len() / 2], b""] {
        let r = call(p2, "PUT", "/api/model.gguf", bad);
        assert_eq!(r.status, 400, "{}", r.text());
        assert!(field(&r.json(), "error")
            .as_str()
            .unwrap()
            .contains("usable"));
    }
    srv.stop();
    srv2.stop();
}

#[test]
fn a_model_with_another_vocabulary_loads_but_refuses_text() {
    use ferrum_model::gguf::gpt2::Quant;
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let mut cfg = ferrum_gpu_gui::engine::preset("tiny").unwrap();
    cfg.vocab = 300;
    let m = ferrum_ref::llama::Llama::random(cfg, 2, false, false);
    let bytes = ferrum_model::gguf::llama::to_gguf(&m, Quant::F32);
    let up = call(p, "PUT", "/api/model.gguf", &bytes);
    assert_eq!(up.status, 200);
    assert_eq!(field(&up.json(), "byte_tokens"), Json::Bool(false));
    let g = post(p, "/api/generate", r#"{"prompt":"hi"}"#);
    assert_eq!(g.status, 400);
    assert!(g.text().contains("byte-level"));
    srv.stop();
}
