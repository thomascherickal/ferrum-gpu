//! Start the server on loopback around any [`Decoder`].

use crate::decoder::Decoder;
use crate::routes::{route, MAX_BODY};
use ferrum_gpu_gui::http::{serve, Handler};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// A running server.
pub struct Running {
    /// Where it listens.
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Running {
    /// Stop accepting connections and wait for the accept loop to end.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Listen on `127.0.0.1:port` (`0` = any free port).
pub fn start(d: Arc<dyn Decoder>, port: u16) -> Result<Running, String> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let handler: Handler = Arc::new(move |req| route(d.as_ref(), req));
    let s = stop.clone();
    let thread = std::thread::spawn(move || serve(listener, handler, s, MAX_BODY));
    Ok(Running {
        addr,
        stop,
        thread: Some(thread),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::StubDecoder;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    fn http(addr: SocketAddr, method: &str, path: &str, body: &str) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        write!(
            s,
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            addr.port(),
            body.len()
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn serves_over_real_sockets_and_refuses_foreign_hosts() {
        let r = start(Arc::new(StubDecoder), 0).unwrap();
        assert!(
            http(r.addr, "GET", "/health", "").contains("200")
                && http(r.addr, "GET", "/health", "").contains("ok")
        );
        let g = http(
            r.addr,
            "POST",
            "/v1/generate",
            r#"{"prompt":"x","max_new":2}"#,
        );
        assert!(g.contains("!!"), "{g}");
        let mut s = TcpStream::connect(r.addr).unwrap();
        write!(
            s,
            "GET /health HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert!(out.contains("403"), "{out}");
        r.stop();
    }

    #[test]
    fn a_taken_port_is_an_error() {
        let a = start(Arc::new(StubDecoder), 0).unwrap();
        let e = start(Arc::new(StubDecoder), a.addr.port())
            .err()
            .expect("port taken");
        assert!(e.contains("cannot listen"));
        a.stop();
    }
}
