//! Ollama: `export-ollama` registers a model with a real `ollama serve`, and Ollama's raw greedy
//! text equals ours. The server gets a private HOME and model directory (it writes an identity key
//! and its blobs there, never into the user's own) and is kept off the GPU.

use crate::skip;
use crate::support::model::{ferrum, fixture, ours, NEW_TOKENS, PROMPT};
use crate::support::proc::{wait_until, with_deadline, Group};
use crate::support::tools::ollama;
use ferrum_gpu_gui::json::Json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A running `ollama serve`, ended with its process group when dropped.
struct Server {
    _group: Group,
    host: String,
    home: PathBuf,
    bin: PathBuf,
}

impl Server {
    /// The environment every Ollama client and server here shares.
    fn env(&self) -> Vec<(&'static str, String)> {
        vec![
            ("HOME", self.home.display().to_string()),
            ("OLLAMA_HOST", self.host.clone()),
            (
                "OLLAMA_MODELS",
                self.home.join("models").display().to_string(),
            ),
            ("CUDA_VISIBLE_DEVICES", "-1".into()),
            ("HIP_VISIBLE_DEVICES", "-1".into()),
            ("OLLAMA_LLM_LIBRARY", "cpu".into()),
        ]
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Start a server for test `tag`; `None` if it never became ready.
fn start(bin: &Path, tag: &str) -> Option<Server> {
    let home = fixture().dir.join(format!("ollama-{tag}"));
    std::fs::create_dir_all(home.join("models")).unwrap();
    let host = format!("127.0.0.1:{}", free_port());
    let mut s = Server {
        _group: Group::spawn(&mut Command::new("true")),
        host: host.clone(),
        home,
        bin: bin.to_path_buf(),
    };
    let mut cmd = with_deadline(bin, 900);
    cmd.arg("serve");
    for (k, v) in s.env() {
        cmd.env(k, v);
    }
    s._group = Group::spawn(&mut cmd);
    wait_until(40, || {
        http(&host, "GET", "/api/version", "").is_some_and(|(c, _)| c == 200)
    })
    .then_some(s)
}

/// One HTTP request; the status and the (de-chunked) body, or `None` if the connection failed.
fn http(host: &str, method: &str, path: &str, body: &str) -> Option<(u16, String)> {
    let mut c = TcpStream::connect(host).ok()?;
    c.set_read_timeout(Some(std::time::Duration::from_secs(120)))
        .ok()?;
    write!(
        c,
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut raw = Vec::new();
    c.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n")?;
    let status = head.split(' ').nth(1)?.parse().ok()?;
    let body = if head.to_lowercase().contains("transfer-encoding: chunked") {
        dechunk(body)
    } else {
        body.to_string()
    };
    Some((status, body))
}

fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((len, rest)) = s.split_once("\r\n") {
        let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
        if n == 0 || rest.len() < n {
            break;
        }
        out.push_str(&rest[..n]);
        s = rest[n..].trim_start_matches("\r\n");
    }
    out
}

/// Register `model` as `name` through `ferrum-gpu export-ollama` against `server`.
fn export(server: &Server, model: &Path, name: &str) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"));
    cmd.args(["export-ollama", "--model"])
        .arg(model)
        .args(["--name", name, "--bin"])
        .arg(&server.bin);
    for (k, v) in server.env() {
        cmd.env(k, v);
    }
    cmd.output().expect("run export-ollama")
}

/// Ollama's raw greedy continuation of [`PROMPT`] with the registered `name`.
fn raw_generate(server: &Server, name: &str) -> String {
    let body = format!(
        r#"{{"model":"{name}","prompt":"{PROMPT}","raw":true,"stream":false,"options":{{"temperature":0,"seed":1,"num_predict":{NEW_TOKENS},"num_gpu":0}}}}"#
    );
    let (code, reply) = http(&server.host, "POST", "/api/generate", &body).expect("reach ollama");
    assert_eq!(code, 200, "{reply}");
    let j = Json::parse(&reply).unwrap_or_else(|e| panic!("{e}: {reply}"));
    j.get("response")
        .and_then(Json::as_str)
        .expect("a response")
        .to_string()
}

#[test]
fn export_ollama_registers_the_model_and_ollama_continues_it_like_we_do() {
    let Some(bin) = ollama() else {
        skip!("ollama not installed (set FERRUM_OLLAMA)")
    };
    let Some(server) = start(&bin, "run") else {
        skip!("ollama serve did not start")
    };
    let f = fixture();
    for (label, path) in [("f32", &f.f32), ("q8_0", &f.q8)] {
        let name = format!("ferrum-interop-{label}");
        let o = export(&server, path, &name);
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
        let theirs = raw_generate(&server, &name);
        let mine = ours(path);
        assert_eq!(format!("{PROMPT}{theirs}").trim_end(), mine, "{label}");
    }
}

#[test]
fn ollama_lists_and_describes_the_registered_model() {
    let Some(bin) = ollama() else {
        skip!("ollama not installed (set FERRUM_OLLAMA)")
    };
    let Some(server) = start(&bin, "show") else {
        skip!("ollama serve did not start")
    };
    let f = fixture();
    assert!(export(&server, &f.f32, "ferrum-interop-show")
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
