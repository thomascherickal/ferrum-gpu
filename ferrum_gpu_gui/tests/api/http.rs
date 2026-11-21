use ferrum_gpu_gui::json::Json;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub(super) struct Reply {
    pub(super) status: u16,
    pub(super) head: String,
    pub(super) body: Vec<u8>,
}

impl Reply {
    pub(super) fn json(&self) -> Json {
        Json::parse(std::str::from_utf8(&self.body).expect("utf-8"))
            .unwrap_or_else(|e| panic!("{e}: {:?}", String::from_utf8_lossy(&self.body)))
    }
    pub(super) fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

pub(super) fn raw(port: u16, request: &[u8]) -> Reply {
    let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    c.write_all(request).unwrap();
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    let split = out
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a response head");
    let head = String::from_utf8_lossy(&out[..split]).into_owned();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    Reply {
        status,
        head,
        body: out[split + 4..].to_vec(),
    }
}

pub(super) fn call(port: u16, method: &str, path: &str, body: &[u8]) -> Reply {
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    req.extend_from_slice(body);
    raw(port, &req)
}

pub(super) fn post(port: u16, path: &str, body: &str) -> Reply {
    call(port, "POST", path, body.as_bytes())
}

pub(super) fn get(port: u16, path: &str) -> Reply {
    call(port, "GET", path, b"")
}
