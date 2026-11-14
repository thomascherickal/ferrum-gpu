//! A real `ollama serve` for the tests: a private port, a private `HOME` and model directory (it
//! writes an identity key and its blobs there, never into the user's own), in its own process group.

use super::model::{fixture, NEW_TOKENS, PROMPT};
use super::proc::{wait_until, with_deadline, Group};
use ferrum_gpu_gui::json::Json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Which hardware the server may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// No GPU: `CUDA_VISIBLE_DEVICES=-1` and the CPU runner.
    Cpu,
    /// The GPU through CUDA (Ollama's default choice).
    Cuda,
    /// The GPU through Vulkan (`OLLAMA_VULKAN=1`, CUDA hidden).
    Vulkan,
}

/// A running server, ended with its process group when dropped.
pub struct Server {
    _group: Group,
    pub host: String,
    pub home: PathBuf,
    pub bin: PathBuf,
    pub mode: Mode,
    pub log: PathBuf,
}

impl Server {
    /// The environment the server and every client of it share.
    pub fn env(&self) -> Vec<(&'static str, String)> {
        let mut e = vec![
            ("HOME", self.home.display().to_string()),
            ("OLLAMA_HOST", self.host.clone()),
            (
                "OLLAMA_MODELS",
                self.home.join("models").display().to_string(),
            ),
            ("OLLAMA_DEBUG", "1".into()),
        ];
        match self.mode {
            Mode::Cpu => {
                e.push(("CUDA_VISIBLE_DEVICES", "-1".into()));
                e.push(("HIP_VISIBLE_DEVICES", "-1".into()));
                e.push(("OLLAMA_LLM_LIBRARY", "cpu".into()));
            }
            Mode::Cuda => {}
            Mode::Vulkan => {
                e.push(("OLLAMA_VULKAN", "1".into()));
                e.push(("CUDA_VISIBLE_DEVICES", "-1".into()));
            }
        }
        e
    }

    /// Register `model` as `name` through `ferrum-gpu export-ollama` against this server.
    pub fn export(&self, model: &Path, name: &str) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"));
        cmd.args(["export-ollama", "--model"])
            .arg(model)
            .args(["--name", name, "--bin"])
            .arg(&self.bin);
        for (k, v) in self.env() {
            cmd.env(k, v);
        }
        cmd.output().expect("run export-ollama")
    }

    /// Ollama's raw greedy continuation of the prompt with the registered `name`. `cpu_only` adds
    /// `num_gpu: 0` to the request; the GPU tests leave the choice to Ollama.
    pub fn raw_generate(&self, name: &str, cpu_only: bool) -> String {
        let gpu = if cpu_only { r#","num_gpu":0"# } else { "" };
        let body = format!(
            r#"{{"model":"{name}","prompt":"{PROMPT}","raw":true,"stream":false,"options":{{"temperature":0,"seed":1,"num_predict":{NEW_TOKENS},"num_thread":1{gpu}}}}}"#
        );
        let (code, reply) = http(&self.host, "POST", "/api/generate", &body).expect("reach ollama");
        assert_eq!(code, 200, "{reply}");
        let j = Json::parse(&reply).unwrap_or_else(|e| panic!("{e}: {reply}"));
        j.get("response")
            .and_then(Json::as_str)
            .expect("a response")
            .to_string()
    }

    /// `(size, size_vram)` in bytes of the loaded model `name`, from `GET /api/ps`.
    pub fn loaded_sizes(&self, name: &str) -> Option<(u64, u64)> {
        let (_, body) = http(&self.host, "GET", "/api/ps", "")?;
        let j = Json::parse(&body).ok()?;
        let Some(Json::Arr(models)) = j.get("models") else {
            return None;
        };
        let m = models.iter().find(|m| {
            m.get("name")
                .and_then(Json::as_str)
                .is_some_and(|n| n.starts_with(name))
        })?;
        let size = |k: &str| m.get(k).and_then(Json::as_f64).map(|v| v as u64);
        Some((size("size")?, size("size_vram")?))
    }

    /// The server's own log so far.
    pub fn log_text(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }
}

pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Start a server for test `tag`; `None` if it never became ready. Its output goes to a log file.
pub fn start(bin: &Path, tag: &str, mode: Mode) -> Option<Server> {
    let home = fixture().dir.join(format!("ollama-{tag}"));
    std::fs::create_dir_all(home.join("models")).unwrap();
    let host = format!("127.0.0.1:{}", free_port());
    let log = home.join("server.log");
    let mut s = Server {
        _group: Group::spawn(&mut Command::new("true")),
        host: host.clone(),
        home,
        bin: bin.to_path_buf(),
        mode,
        log: log.clone(),
    };
    let mut cmd = with_deadline(bin, 900);
    cmd.arg("serve");
    for (k, v) in s.env() {
        cmd.env(k, v);
    }
    s._group = Group::spawn_logged(&mut cmd, &log);
    let ready = wait_until(60, || {
        http(&host, "GET", "/api/version", "").is_some_and(|(c, _)| c == 200)
    });
    ready.then_some(s)
}

/// One HTTP request; the status and the (de-chunked) body, or `None` if the connection failed.
pub fn http(host: &str, method: &str, path: &str, body: &str) -> Option<(u16, String)> {
    let mut c = TcpStream::connect(host).ok()?;
    c.set_read_timeout(Some(std::time::Duration::from_secs(300)))
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
