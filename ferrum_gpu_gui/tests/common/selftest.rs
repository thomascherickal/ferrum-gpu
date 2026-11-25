use std::io::{Read, Write};
use std::time::Duration;

/// The body of `GET /api/selftest/result` from the server on `port`, once the page has posted
/// one (a 200 reply); `None` while it has not or the server cannot be reached.
pub(crate) fn selftest_result(port: u16) -> Option<String> {
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    c.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        c,
        "GET /api/selftest/result HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n"
    )
    .ok()?;
    let mut out = String::new();
    c.read_to_string(&mut out).ok()?;
    let (head, body) = out.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.1 200").then(|| body.to_string())
}
