use super::*;
use crate::http::{MAX_HEAD, MAX_HEADERS};
use crate::serving::limits::Limits;
use ferrum_test_support::load::ask;

/// The parser's limits, seen through the real JSON server: each hostile request gets its status
/// and message, and the server keeps answering.
#[test]
fn the_request_limits_hold_over_a_real_socket() {
    let cfg = Config {
        token: Some("s3cret".into()),
        limits: Limits {
            max_body: 64,
            ..Limits::default()
        },
        ..Config::default()
    };
    let r = start(Arc::new(StubDecoder), &cfg).unwrap();
    let (addr, auth) = (r.addr, "Authorization: Bearer s3cret\r\n");
    let head = |extra: &str| format!("POST /v1/generate HTTP/1.1\r\nHost: {addr}\r\n{auth}{extra}");
    let many: String = (0..=MAX_HEADERS).map(|i| format!("X-{i}: v\r\n")).collect();
    let cases = [
        (head(&format!("X: {}\r\n\r\n", "a".repeat(MAX_HEAD))), 431, "request head too large"),
        (head(&format!("{many}\r\n")), 431, "too many headers"),
        (head("Content-Length: 65\r\n\r\n"), 413, "request body too large"),
        (head("Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"), 501, "chunked"),
        (head("Content-Length: 1\r\nContent-Length: 2\r\n\r\nab"), 400, "conflicting Content-Length"),
        (head("\r\n"), 411, "Content-Length required"),
        (format!("GET /v1/info%00 HTTP/1.1\r\nHost: {addr}\r\n{auth}\r\n"), 404, ""),
        (format!("GET /health HTTP/1.1\r\nHost: localhost.evil.example:{}\r\n\r\n", addr.port()), 403, "Host"),
        (format!("GET /health HTTP/1.1\r\nHost: {addr}.nip.io\r\n\r\n"), 403, "Host"),
        (format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nOrigin: http://{addr}.evil.example\r\n\r\n"), 403, "cross-origin"),
        (format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nOrigin: null\r\n\r\n"), 403, "cross-origin"),
        (format!("GET /v1/info HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer s3cre\r\n\r\n"), 401, "bearer"),
        (format!("GET /v1/info HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer s3cret0\r\n\r\n"), 401, "bearer"),
        (format!("GET /v1/info HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer \r\n\r\n"), 401, "bearer"),
    ];
    for (raw, status, message) in cases {
        let a = ask(addr, raw.as_bytes());
        assert_eq!(a.status, status, "{raw:.120}: {}", a.body);
        assert!(a.body.contains(message), "{raw:.120}: {}", a.body);
    }
    let ok = format!("GET /v1/info HTTP/1.1\r\nHost: {addr}\r\n{auth}\r\n");
    assert_eq!(ask(addr, ok.as_bytes()).status, 200);
    r.stop();
}
