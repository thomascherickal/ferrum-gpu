//! Spawning the binary and talking to it over a socket.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

/// The binary with `args`, stdout piped, stderr piped.
pub(crate) fn spawn(args: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_ferrum_gpu_gui"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start ferrum_gpu_gui")
}

/// A running server that is killed (by its own pid) when dropped.
pub(crate) struct Server {
    pub child: Child,
    pub port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

/// Start the binary on a free port with the CPU backend, read its first lines for the URL, and
/// stop reading (the pipe closes when the reader is dropped).
pub(crate) fn serve() -> Server {
    let mut child = spawn(&["--port", "0", "--device", "cpu", "--threads", "1"]);
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let port = loop {
        let l = lines.next().expect("the binary exited early").unwrap();
        if let Some(u) = l.strip_prefix("ferrum-gpu GUI: http://127.0.0.1:") {
            break u.trim_end_matches('/').parse().expect("a port");
        }
    };
    Server { child, port }
}

/// `GET path` with a correct `Host`; the whole response as text.
pub(crate) fn get(port: u16, path: &str) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

/// Run to completion; exit code, stdout and stderr.
pub(crate) fn finish(args: &[&str]) -> (Option<i32>, String, String) {
    let o = spawn(args).wait_with_output().unwrap();
    (
        o.status.code(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}
