use super::*;
use crate::serving::decoder::StubDecoder;
use std::io::{Read, Write};
use std::net::TcpStream;

fn http(addr: SocketAddr, method: &str, path: &str, extra: &str, body: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        addr.port(),
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

/// A decoder that "generates" one token every 5 ms until cancelled (or 20 s), and records how it
/// ended, so a test can see that a disconnect frees it.
struct SlowDecoder {
    tokens: std::sync::atomic::AtomicUsize,
    cancelled: AtomicBool,
}

impl Decoder for SlowDecoder {
    fn info(&self) -> crate::serving::decoder::Info {
        StubDecoder.info()
    }
    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        StubDecoder.encode(text)
    }
    fn bos(&self) -> usize {
        0
    }
    fn complete(
        &self,
        _prompt: &str,
        _r: &crate::serving::decoder::GenReq,
    ) -> Result<(String, usize), String> {
        for _ in 0..4000 {
            if let Err(e) = crate::cancel::check() {
                self.cancelled.store(true, Ordering::SeqCst);
                return Err(e);
            }
            self.tokens.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Ok(("x".into(), 1))
    }
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        StubDecoder.score(ids)
    }
}

mod a_client_that_leaves_stops_the_generation_and_frees_its_slot;
mod loopback_detection_and_the_exposure_rule;
mod serves_over_real_sockets_and_refuses_foreign_hosts;
