use super::*;

#[test]
fn export_ollama_registers_the_model_and_ollama_continues_it_like_we_do() {
    let Some(bin) = ollama() else {
        skip!("ollama not installed (set FERRUM_OLLAMA)")
    };
    let Some(server) = start(&bin, "run", Mode::Cpu) else {
        skip!("ollama serve did not start")
    };
    let f = fixture();
    for (label, path) in [("f32", &f.f32), ("q8_0", &f.q8)] {
        let name = format!("ferrum-interop-{label}");
        let o = server.export(path, &name);
        let text = String::from_utf8_lossy(&o.stdout);
        assert!(
            o.status.success(),
            "{label}: {text}{}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(
            text.contains(&format!("created Ollama model '{name}'")),
            "{text}"
        );
        let theirs = server.raw_generate(&name, true);
        let mine = ours(path);
        assert_eq!(format!("{PROMPT}{theirs}").trim_end(), mine, "{label}");
    }
}

#[test]
fn ollama_lists_and_describes_the_registered_model() {
    let Some(bin) = ollama() else {
        skip!("ollama not installed (set FERRUM_OLLAMA)")
    };
    let Some(server) = start(&bin, "show", Mode::Cpu) else {
        skip!("ollama serve did not start")
    };
    let f = fixture();
    assert!(server
        .export(&f.f32, "ferrum-interop-show")
        .status
        .success());
    let (code, tags) = http(&server.host, "GET", "/api/tags", "").unwrap();
    assert_eq!(code, 200);
    assert!(tags.contains("ferrum-interop-show"), "{tags}");
    let (code, show) = http(
        &server.host,
        "POST",
        "/api/show",
        r#"{"model":"ferrum-interop-show"}"#,
    )
    .unwrap();
    assert_eq!(code, 200, "{show}");
    let j = Json::parse(&show).unwrap();
    let family = j
        .get("details")
        .and_then(|d| d.get("family"))
        .and_then(Json::as_str);
    assert_eq!(family, Some("gpt2"), "{show}");
    // our Modelfile's parameters came through
    let params = j.get("parameters").and_then(Json::as_str).unwrap_or("");
    assert!(
        params.contains("num_ctx") && params.contains("16"),
        "{params}"
    );
    assert!(params.contains("<|endoftext|>"), "{params}");
}
