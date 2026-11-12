use super::*;
use crate::decoder::StubDecoder;
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

#[test]
fn serves_over_real_sockets_and_refuses_foreign_hosts() {
    let r = start(Arc::new(StubDecoder), &Config::default()).unwrap();
    let h = http(r.addr, "GET", "/health", "", "");
    assert!(h.contains("200") && h.contains("ok"));
    let g = http(
        r.addr,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"x","max_tokens":2}]}"#,
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
fn the_token_gates_everything_but_health() {
    let cfg = Config {
        token: Some("s3cret".into()),
        ..Config::default()
    };
    let r = start(Arc::new(StubDecoder), &cfg).unwrap();
    assert!(http(r.addr, "GET", "/health", "", "").contains("200"));
    let denied = http(r.addr, "GET", "/v1/info", "", "");
    assert!(
        denied.contains("401") && denied.contains("WWW-Authenticate: Bearer"),
        "{denied}"
    );
    assert!(http(
        r.addr,
        "GET",
        "/v1/info",
        "Authorization: Bearer wrong\r\n",
        ""
    )
    .contains("401"));
    assert!(http(
        r.addr,
        "GET",
        "/v1/info",
        "Authorization: Bearer s3cret\r\n",
        ""
    )
    .contains("200"));
    r.stop();
}

#[test]
fn a_taken_port_is_an_error() {
    let a = start(Arc::new(StubDecoder), &Config::default()).unwrap();
    let e = start(
        Arc::new(StubDecoder),
        &Config {
            port: a.addr.port(),
            ..Config::default()
        },
    )
    .err()
    .expect("port taken");
    assert!(e.contains("cannot listen"));
    a.stop();
}

#[test]
fn loopback_detection_and_the_exposure_rule() {
    for h in ["127.0.0.1", "::1", "localhost", "127.5.5.5"] {
        assert!(is_loopback(h), "{h}");
        assert!(check_exposure(h, None, false).is_ok());
    }
    for h in ["0.0.0.0", "192.168.1.5", "example.org"] {
        assert!(!is_loopback(h), "{h}");
        let e = check_exposure(h, None, false).unwrap_err();
        assert!(
            e.contains("--token") && e.contains("--allow-unauthenticated"),
            "{e}"
        );
        assert!(check_exposure(h, Some("t"), false).is_ok());
        assert!(check_exposure(h, None, true).is_ok());
    }
}

#[test]
fn the_in_flight_cap_refuses_with_503_and_frees_slots() {
    let n = AtomicUsize::new(0);
    let slots: Vec<_> = (0..MAX_INFLIGHT)
        .map(|_| Slot::take(&n).expect("slot"))
        .collect();
    assert!(Slot::take(&n).is_none(), "the 33rd is refused");
    assert_eq!(
        n.load(Ordering::SeqCst),
        MAX_INFLIGHT,
        "a refusal gives its count back"
    );
    drop(slots);
    assert_eq!(n.load(Ordering::SeqCst), 0);
    assert!(Slot::take(&n).is_some());
}

#[test]
fn a_non_loopback_server_with_a_token_skips_the_host_check() {
    let cfg = Config {
        host: "0.0.0.0".into(),
        token: Some("t".into()),
        ..Config::default()
    };
    let r = start(Arc::new(StubDecoder), &cfg).unwrap();
    let mut s = TcpStream::connect(("127.0.0.1", r.addr.port())).unwrap();
    write!(s, "GET /v1/info HTTP/1.1\r\nHost: my.server.example\r\nAuthorization: Bearer t\r\nConnection: close\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.contains("200"), "{out}");
    r.stop();
}

/// A decoder that "generates" one token every 5 ms until cancelled (or 20 s), and records how it
/// ended, so a test can see that a disconnect frees it.
struct SlowDecoder {
    tokens: std::sync::atomic::AtomicUsize,
    cancelled: AtomicBool,
}

impl Decoder for SlowDecoder {
    fn info(&self) -> crate::decoder::Info {
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
        _r: &crate::decoder::GenReq,
    ) -> Result<(String, usize), String> {
        for _ in 0..4000 {
            if let Err(e) = ferrum_gpu_gui::cancel::check() {
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

#[test]
fn a_client_that_leaves_stops_the_generation_and_frees_its_slot() {
    let d = Arc::new(SlowDecoder {
        tokens: 0.into(),
        cancelled: AtomicBool::new(false),
    });
    let r = start(d.clone(), &Config::default()).unwrap();
    let body = r#"{"requests":[{"prompt":"x","max_tokens":2}]}"#;
    let mut s = TcpStream::connect(r.addr).unwrap();
    write!(
        s,
        "POST /v1/generate HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Length: {}\r\n\r\n{body}",
        r.addr.port(),
        body.len()
    )
    .unwrap();
    while d.tokens.load(Ordering::SeqCst) < 5 {
        std::thread::sleep(std::time::Duration::from_millis(5)); // it is generating
    }
    drop(s); // the client goes away
    let t0 = std::time::Instant::now();
    while !d.cancelled.load(Ordering::SeqCst) {
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "still generating {} tokens after the client left",
            d.tokens.load(Ordering::SeqCst)
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let n = d.tokens.load(Ordering::SeqCst);
    assert!(n < 1000, "stopped long before the 4000-token budget: {n}");
    r.stop();
}
