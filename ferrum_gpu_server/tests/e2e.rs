//! End-to-end: the real `ferrum-gpu-server` binary on a tiny model, queried over HTTP.

use ferrum_gpu_gui::engine::fresh_model;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::to_gguf;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

fn http(addr: &str, method: &str, path: &str, body: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn the_binary_serves_all_four_routes_on_the_cpu() {
    let dir = std::env::temp_dir().join(format!("ferrum_srv_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--device",
            "cpu",
            "--port",
            "0",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let _server = Server(child);
    let addr = loop {
        let l = lines.next().expect("server exited early").unwrap();
        if let Some(u) = l.strip_prefix("ferrum-gpu server: http://") {
            break u.trim_end_matches('/').to_string();
        }
    };
    assert!(http(&addr, "GET", "/health", "").contains(r#""status":"ok""#));
    let info = http(&addr, "GET", "/v1/info", "");
    assert!(
        info.contains(r#""kind":"cpu""#) && info.contains(r#""vocab":256"#),
        "{info}"
    );
    let g = http(
        &addr,
        "POST",
        "/v1/generate",
        r#"{"prompt":"hi","max_new":5}"#,
    );
    assert!(g.contains("200") && g.contains(r#""tokens":5"#), "{g}");
    let l = http(
        &addr,
        "POST",
        "/v1/loglikelihood",
        r#"{"context":"hello","continuation":" world"}"#,
    );
    assert!(
        l.contains("200") && l.contains("loglikelihood") && l.contains(r#""tokens":6"#),
        "{l}"
    );
    assert!(http(&addr, "POST", "/v1/generate", "oops").contains("400"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_missing_model_exits_with_an_error() {
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .args(["--model", "/nope.gguf", "--device", "cpu"])
        .output()
        .unwrap();
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("/nope.gguf"));
}
