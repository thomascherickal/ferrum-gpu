//! Drives the Studio tabs in headless Chrome against the real server: keyboard tab movement, Beginner/Expert, tool-tips, Datasets, the file picker, and real
//! Prepare and Tokenize jobs (the real commands, in-process). Skipped (with the reason printed) when
//! Chrome is missing, so the suite passes on a bare machine.

// the harness uses process groups, `kill`, `sh` and `timeout`: not built on Windows
#![cfg(unix)]

mod common;

use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

fn fetch_result(port: u16) -> Option<String> {
    use std::io::{Read, Write};
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    c.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        c,
        "GET /api/selftest/result HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n"
    )
    .ok()?;
    let mut out = String::new();
    c.read_to_string(&mut out).ok()?;
    let (head, body) = out.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.1 200").then(|| body.to_string())
}

fn work(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_studio_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// 30 short distinct documents in `dir/in`, and `dir/corpus.txt` with all of them.
fn corpus(dir: &Path) -> String {
    let inp = dir.join("in");
    std::fs::create_dir_all(&inp).unwrap();
    let mut all = String::new();
    for i in 0..30 {
        let doc =
            format!("document {i}: the quick brown fox jumps over the lazy dog, again and again. ");
        let doc = doc.repeat(8);
        std::fs::write(inp.join(format!("d{i:02}.txt")), &doc).unwrap();
        all.push_str(&doc);
    }
    std::fs::write(dir.join("corpus.txt"), &all).unwrap();
    inp.display().to_string()
}

#[test]
fn the_new_tabs_work_in_a_real_browser_with_the_real_commands() {
    let Some(chrome) = common::chrome() else {
        eprintln!("SKIP: no Chrome or Chromium installed");
        return;
    };
    let dir = work("tabs");
    let inputs = corpus(&dir);
    // a tokenizer file, as a training run saves it
    let tok = dir.join("tok.gguf");
    let train: Vec<String> = [
        "train", "--preset", "tiny", "--steps", "2", "--batch", "2", "--vocab", "300", "--device",
        "cpu", "--data",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([
        dir.join("corpus.txt").display().to_string(),
        "--out".into(),
        dir.join("m.gguf").display().to_string(),
        "--tokenizer".into(),
        tok.display().to_string(),
    ])
    .collect();
    ferrum_gpu_core::commands::run(&train, &mut Vec::new()).unwrap();
    assert!(tok.is_file());

    let app = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let server = start(app, 0).unwrap();
    let port = server.addr.port();
    let out = dir.join("prepared");
    let tokout = dir.join("tokens.gguf");
    let gtout = dir.join("trained.gguf");
    let gtstop = dir.join("stopped.gguf");
    let ftout = dir.join("tuned.gguf");
    let q = |k: &str, v: &Path| format!("{k}={}", v.display());
    let url = format!(
        "http://127.0.0.1:{port}/?autotest=studio&{}&{}&{}&{}&{}&{}&{}&{}&{}",
        q("file", &dir.join("corpus.txt")),
        q("dir", &dir),
        format_args!("inputs={inputs}"),
        q("out", &out),
        q("tok", &tok),
        q("tokout", &tokout),
        q("gtout", &gtout),
        q("gtstop", &gtstop),
        q("ftout", &ftout),
    );
    let browser = common::Browser::open(&chrome, &url, "studio", 240);
    let Some(result) = common::wait_for(Duration::from_secs(150), || fetch_result(port)) else {
        panic!(
            "the page never reported a result; browser console:\n{}",
            browser.console()
        );
    };
    eprintln!("page reported: {result}");
    let j = Json::parse(&result).unwrap();
    assert_eq!(j.get("kind").and_then(Json::as_str), Some("studio"));
    assert_eq!(j.get("pass"), Some(&Json::Bool(true)), "{result}");
    for f in ["train.txt", "val.txt", "test.txt", "manifest.json"] {
        assert!(out.join(f).is_file(), "prepare wrote {f}");
    }
    assert!(
        tokout.metadata().unwrap().len() > 0,
        "tokenize wrote the token file"
    );
    assert!(
        gtout.metadata().unwrap().len() > 0,
        "the Train tab saved a model"
    );
    assert!(
        gtstop.metadata().unwrap().len() > 0,
        "a stopped run still saved the model it had"
    );
    assert!(
        ftout.metadata().unwrap().len() > 0,
        "the Fine-tune tab saved a model"
    );
    server.stop();
}
