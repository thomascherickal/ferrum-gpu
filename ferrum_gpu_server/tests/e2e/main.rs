//! End-to-end: the real `ferrum-gpu-server` binary on tiny models, queried over HTTP.
use ferrum_gpu_core::engine::fresh_model;
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

mod a_gpt2_file_serves_with_its_bpe_tokenizer;
mod a_llama_style_file_serves_all_five_routes;
mod the_server_keeps_serving_after_its_stdout_is_closed;
mod with_a_serve_time_the_server_exits_normally;
