//! End-to-end: the real `ferrum-gpu-server` binary on tiny models, queried over HTTP.

use ferrum_gpu_gui::engine::fresh_model;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::to_gguf;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

struct Server {
    child: Child,
    addr: String,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

fn http(addr: &str, method: &str, path: &str, extra: &str, body: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

fn work(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_srv_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn spawn(model: &str, extra: &[&str], env: &[(&str, &str)]) -> Server {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"));
    cmd.args(["--model", model, "--device", "cpu", "--port", "0"])
        .args(extra)
        .stdout(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let addr = loop {
        let l = lines.next().expect("server exited early").unwrap();
        if let Some(u) = l.strip_prefix("ferrum-gpu server: http://") {
            break u.trim_end_matches('/').to_string();
        }
    };
    Server { child, addr }
}

#[test]
fn a_llama_style_file_serves_all_five_routes() {
    let dir = work("llama");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let s = spawn(model.to_str().unwrap(), &[], &[]);
    let a = &s.addr;
    assert!(http(a, "GET", "/health", "", "").contains(r#""status":"ok""#));
    let info = http(a, "GET", "/v1/info", "", "");
    assert!(
        info.contains(r#""kind":"cpu""#)
            && info.contains(r#""vocab":256"#)
            && info.contains(r#""arch":"llama""#),
        "{info}"
    );
    let g = http(
        a,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"hi","max_tokens":5}]}"#,
    );
    assert!(g.contains("200") && g.contains(r#""n_tokens":5"#), "{g}");
    let l = http(
        a,
        "POST",
        "/v1/loglikelihood",
        "",
        r#"{"requests":[{"context":"hello","continuation":" world"}]}"#,
    );
    assert!(
        l.contains("200") && l.contains("logprob") && l.contains(r#""n_tokens":6"#),
        "{l}"
    );
    let r = http(
        a,
        "POST",
        "/v1/loglikelihood_rolling",
        "",
        r#"{"texts":["hello world"]}"#,
    );
    assert!(r.contains("200") && r.contains(r#""n_tokens":11"#), "{r}");
    assert!(http(a, "POST", "/v1/generate", "", "oops").contains("400"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_gpt2_file_serves_with_its_bpe_tokenizer() {
    let dir = work("gpt2");
    let (data, model) = (dir.join("c.txt"), dir.join("m.gguf"));
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    let cmd: Vec<String> = format!(
        "train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 8 --epochs 10 --vocab 300 --data {} --out {}",
        data.display(),
        model.display()
    )
    .split_whitespace()
    .map(String::from)
    .collect();
    ferrum_gpu_cli::cmds::run(&cmd, &mut Vec::new()).unwrap();
    let s = spawn(model.to_str().unwrap(), &[], &[]);
    let a = &s.addr;
    let info = http(a, "GET", "/v1/info", "", "");
    assert!(
        info.contains(r#""arch":"gpt2""#) && info.contains(r#""vocab":269"#),
        "{info}"
    );
    let g = http(
        a,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"the c","max_tokens":6,"stop":["zzz"]}]}"#,
    );
    assert!(g.contains("200") && g.contains(r#""text":""#), "{g}");
    let l = http(
        a,
        "POST",
        "/v1/loglikelihood",
        "",
        r#"{"requests":[{"context":"the cat sat","continuation":" on the mat."}]}"#,
    );
    let n = l
        .split(r#""n_tokens":"#)
        .nth(1)
        .and_then(|x| x.split([',', '}']).next())
        .and_then(|x| x.parse::<usize>().ok());
    assert!(
        l.contains("200") && n.is_some_and(|n| n > 0 && n < 12),
        "BPE should use fewer tokens than characters: {l}"
    );
    assert!(http(
        a,
        "POST",
        "/v1/loglikelihood_rolling",
        "",
        r#"{"texts":["the cat sat on the mat. the cat sat"]}"#
    )
    .contains("200"));
    assert!(http(
        a,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"x","chat":true}]}"#
    )
    .contains("400"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_token_gate_works_through_the_binary_and_the_environment() {
    let dir = work("auth");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    for (extra, env) in [
        (vec!["--token", "s3cret"], vec![]),
        (vec![], vec![("FERRUM_TOKEN", "s3cret")]),
    ] {
        let s = spawn(model.to_str().unwrap(), &extra, &env);
        let a = &s.addr;
        assert!(http(a, "GET", "/health", "", "").contains("200"));
        assert!(http(a, "GET", "/v1/info", "", "").contains("401"));
        assert!(http(a, "GET", "/v1/info", "Authorization: Bearer nope\r\n", "").contains("401"));
        assert!(http(a, "GET", "/v1/info", "Authorization: Bearer s3cret\r\n", "").contains("200"));
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Regression (found by CI): the server printed an `auth:` line after its address line, the test's
/// reader had already stopped, and `println!` panicked on the broken pipe, killing the server.
#[test]
fn the_server_keeps_serving_after_its_stdout_is_closed() {
    let dir = work("closed_stdout");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let s = spawn(model.to_str().unwrap(), &["--token", "s3cret"], &[]);
    // `spawn` returned, so the reader is gone; give the server time to print its remaining lines
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(http(&s.addr, "GET", "/health", "", "").contains("200"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn exposing_the_server_without_a_token_is_refused() {
    let dir = work("expose");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--device",
            "cpu",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
        ])
        .env_remove("FERRUM_TOKEN")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        !o.status.success() && err.contains("not a loopback address") && err.contains("--token"),
        "{err}"
    );
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

#[test]
fn help_prints_usage() {
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(
        o.status.success()
            && String::from_utf8_lossy(&o.stdout).contains("usage: ferrum-gpu-server")
    );
}
