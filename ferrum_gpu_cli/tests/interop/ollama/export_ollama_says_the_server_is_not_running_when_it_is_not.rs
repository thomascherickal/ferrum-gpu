use super::*;

#[test]
fn export_ollama_says_the_server_is_not_running_when_it_is_not() {
    let Some(bin) = ollama() else {
        skip!("ollama not installed (set FERRUM_OLLAMA)")
    };
    let f = fixture();
    // an address nothing listens on
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"))
        .args(["export-ollama", "--model"])
        .arg(&f.f32)
        .args(["--name", "ferrum-interop-none", "--bin"])
        .arg(&bin)
        .env("OLLAMA_HOST", format!("127.0.0.1:{}", free_port()))
        .env("HOME", f.dir.join("ollama-none"))
        .env("USERPROFILE", f.dir.join("ollama-none"))
        .output()
        .unwrap();
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("ollama create") && err.contains("ollama serve"),
        "{err}"
    );
}

#[test]
fn the_modelfile_alone_is_written_without_the_tool() {
    // no Ollama needed: --modelfile-only just writes the text
    let f = fixture();
    let mf = f.dir.join("Modelfile");
    let o = ferrum(&[
        "export-ollama",
        "--model",
        f.f32.to_str().unwrap(),
        "--modelfile-only",
        mf.to_str().unwrap(),
    ]);
    assert!(o.status.success());
    let text = std::fs::read_to_string(&mf).unwrap();
    assert!(
        text.starts_with("FROM \"") && text.contains(&format!("{}\"", f.f32.display())),
        "{text}"
    );
    assert!(text.contains("TEMPLATE \"{{ .Prompt }}\""));
    assert!(
        text.contains("PARAMETER num_ctx 16") && text.contains("PARAMETER stop \"<|endoftext|>\"")
    );
}
