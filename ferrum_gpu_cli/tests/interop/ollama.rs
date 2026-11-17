//! Ollama on the CPU: `export-ollama` registers a model with a real `ollama serve`, and Ollama's raw
//! greedy text equals ours. The server gets a private HOME and model directory (it writes an
//! identity key and its blobs there, never into the user's own) and is kept off the GPU. The GPU
//! runs are in `real/ollama_gpu.rs`.

use crate::skip;
use crate::support::model::{ferrum, fixture, ours, PROMPT};
use crate::support::ollama_server::{free_port, http, start, Mode};
use crate::support::tools::ollama;
use ferrum_gpu_core::json::Json;
use std::process::Command;

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
        text.starts_with("FROM \"/") && text.contains(&format!("{}\"", f.f32.display())),
        "{text}"
    );
    assert!(text.contains("TEMPLATE \"{{ .Prompt }}\""));
    assert!(
        text.contains("PARAMETER num_ctx 16") && text.contains("PARAMETER stop \"<|endoftext|>\"")
    );
}
