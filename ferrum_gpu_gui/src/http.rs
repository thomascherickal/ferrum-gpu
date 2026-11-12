//! A minimal HTTP/1.1 server (std only): one request per connection, `Content-Length` bodies,
//! hard limits on header and body size, read timeouts, and a Host/Origin check against DNS
//! rebinding and cross-site requests (the server only ever listens on loopback).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Largest request head (request line plus headers) accepted.
pub const MAX_HEAD: usize = 16 * 1024;
/// Most headers accepted.
pub const MAX_HEADERS: usize = 64;

/// A parsed request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// `GET`, `POST`, ...
    pub method: String,
    /// Path without the query string, percent-decoded.
    pub path: String,
    /// Decoded query parameters.
    pub query: Vec<(String, String)>,
    /// Header names lower-cased.
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Request {
    /// A header value by lower-case name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// A query parameter by name.
    pub fn param(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A response to send.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// `Content-Type`.
    pub content_type: String,
    /// Extra headers.
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Response {
    /// A response with the given type and body.
    pub fn new(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> Response {
        Response {
            status,
            content_type: content_type.to_string(),
            headers: Vec::new(),
            body: body.into(),
        }
    }

    /// A JSON response.
    pub fn json(status: u16, body: &crate::json::Json) -> Response {
        Response::new(status, "application/json", body.to_string())
    }

    /// A JSON error `{"error": message}`.
    pub fn error(status: u16, message: &str) -> Response {
        Response::json(
            status,
            &crate::json::Json::obj([("error", crate::json::Json::Str(message.into()))]),
        )
    }
}

/// A request that cannot be served, with the status to answer.
#[derive(Debug, PartialEq)]
pub struct HttpError {
    /// Status code.
    pub status: u16,
    /// Why.
    pub message: String,
}

fn bad(status: u16, message: &str) -> HttpError {
    HttpError {
        status,
        message: message.to_string(),
    }
}

/// Reason phrase for the statuses this server emits.
pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

/// Percent-decode `s` (`+` is a space when `plus`); invalid escapes and non-UTF-8 are errors.
pub fn percent_decode(s: &str, plus: bool) -> Result<String, HttpError> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let h = b
                    .get(i + 1..i + 3)
                    .and_then(|h| std::str::from_utf8(h).ok())
                    .and_then(|h| u8::from_str_radix(h, 16).ok())
                    .ok_or_else(|| bad(400, "bad percent escape"))?;
                out.push(h);
                i += 3;
            }
            b'+' if plus => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| bad(400, "request target is not UTF-8"))
}

/// Read one request from `r`, refusing heads over [`MAX_HEAD`] and bodies over `max_body`.
pub fn read_request(r: &mut impl Read, max_body: usize) -> Result<Request, HttpError> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match r.read(&mut byte) {
            Ok(0) => return Err(bad(400, "connection closed before the request head ended")),
            Ok(_) => head.push(byte[0]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return Err(bad(408, "timed out reading the request"))
            }
            Err(_) => return Err(bad(400, "read error")),
        }
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
        if head.len() > MAX_HEAD {
            return Err(bad(431, "request head too large"));
        }
    }
    let text = std::str::from_utf8(&head).map_err(|_| bad(400, "request head is not UTF-8"))?;
    let mut lines = text.split("\r\n");
    let line = lines.next().unwrap_or("");
    let mut parts = line.split(' ');
    let (method, target, version) = (
        parts.next().unwrap_or(""),
        parts.next().unwrap_or(""),
        parts.next().unwrap_or(""),
    );
    if method.is_empty()
        || target.is_empty()
        || parts.next().is_some()
        || !version.starts_with("HTTP/1.")
    {
        return Err(bad(400, "malformed request line"));
    }
    if !method.bytes().all(|c| c.is_ascii_uppercase()) {
        return Err(bad(400, "malformed method"));
    }
    let mut headers: Vec<(String, String)> = Vec::new();
    for l in lines.filter(|l| !l.is_empty()) {
        let (k, v) = l
            .split_once(':')
            .ok_or_else(|| bad(400, "malformed header"))?;
        if k.is_empty() || k.contains(' ') {
            return Err(bad(400, "malformed header name"));
        }
        headers.push((k.to_ascii_lowercase(), v.trim().to_string()));
        if headers.len() > MAX_HEADERS {
            return Err(bad(431, "too many headers"));
        }
    }
    let (raw_path, raw_query) = target.split_once('?').unwrap_or((target, ""));
    if !raw_path.starts_with('/') {
        return Err(bad(400, "request target must be an absolute path"));
    }
    let path = percent_decode(raw_path, false)?;
    let mut query = Vec::new();
    for kv in raw_query.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        query.push((percent_decode(k, true)?, percent_decode(v, true)?));
    }
    let find = |n: &str| headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    if find("transfer-encoding").is_some() {
        return Err(bad(
            501,
            "chunked or encoded request bodies are not supported",
        ));
    }
    let len = match find("content-length") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| bad(400, "bad Content-Length"))?,
        None if matches!(method, "POST" | "PUT") => {
            return Err(bad(411, "Content-Length required"))
        }
        None => 0,
    };
    if len > max_body {
        return Err(bad(413, "request body too large"));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).map_err(|e| {
        if matches!(
            e.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ) {
            bad(408, "timed out reading the body")
        } else {
            bad(400, "body shorter than Content-Length")
        }
    })?;
    Ok(Request {
        method: method.to_string(),
        path,
        query,
        headers,
        body,
    })
}

/// Serialise `resp` onto `w`.
pub fn write_response(w: &mut impl Write, resp: &Response) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n",
        resp.status,
        reason(resp.status),
        resp.content_type,
        resp.body.len()
    );
    for (k, v) in &resp.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(&resp.body)?;
    w.flush()
}

/// `Some(response)` when `req` must be refused: its `Host` is not this loopback server, or it
/// carries a cross-site `Origin`.
pub fn guard(req: &Request, port: u16) -> Option<Response> {
    let ok_hosts = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    match req.header("host") {
        Some(h) if ok_hosts.iter().any(|o| o == h) => {}
        _ => return Some(Response::error(403, "unexpected Host header")),
    }
    if let Some(o) = req.header("origin") {
        if !ok_hosts.iter().any(|h| o == format!("http://{h}")) {
            return Some(Response::error(403, "cross-origin request refused"));
        }
    }
    None
}

/// The handler type the server calls for every accepted request.
pub type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

/// Serve `listener` until `stop` is set: one thread per connection, loopback guard applied
/// before `handler` runs. Returns when the stop flag is seen (the flag is polled between accepts).
pub fn serve(listener: TcpListener, handler: Handler, stop: Arc<AtomicBool>, max_body: usize) {
    serve_with(listener, handler, stop, max_body, true);
}

/// Like [`serve`]; with `loopback_guard` false the Host/Origin check is skipped, for a server
/// that listens beyond loopback and authenticates every request itself.
pub fn serve_with(
    listener: TcpListener,
    handler: Handler,
    stop: Arc<AtomicBool>,
    max_body: usize,
    loopback_guard: bool,
) {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    listener.set_nonblocking(true).ok();
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let h = handler.clone();
                std::thread::spawn(move || handle(stream, &*h, (port, loopback_guard), max_body));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10))
            }
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

fn handle(
    mut stream: TcpStream,
    handler: &(dyn Fn(&Request) -> Response + Send + Sync),
    (port, loopback_guard): (u16, bool),
    max_body: usize,
) {
    stream.set_nonblocking(false).ok();
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
    let resp = match read_request(&mut stream, max_body) {
        Ok(req) => loopback_guard
            .then(|| guard(&req, port))
            .flatten()
            .unwrap_or_else(|| handle_watched(&stream, &req, handler)),
        Err(e) => Response::error(e.status, &e.message),
    };
    let _ = write_response(&mut stream, &resp);
}

/// Run `handler` for `req`; for a POST (the requests that compute), raise this thread's cancel
/// flag ([`crate::cancel`]) as soon as the client closes its end of `stream`.
fn handle_watched(
    stream: &TcpStream,
    req: &Request,
    handler: &(dyn Fn(&Request) -> Response + Send + Sync),
) -> Response {
    let (flag, done) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let watcher = (req.method == "POST")
        .then(|| stream.try_clone().ok())
        .flatten()
        .map(|s| {
            let (f, d) = (flag.clone(), done.clone());
            std::thread::spawn(move || watch_for_close(&s, &f, &d))
        });
    let _scope = crate::cancel::install(flag);
    let resp = handler(req);
    done.store(true, Ordering::Relaxed);
    if let Some(w) = watcher {
        let _ = w.join();
    }
    resp
}

/// Raise `flag` when the peer closes or resets `stream`; return when `done` is set. The request
/// has been read completely, so the socket's read timeout (shared with the handler's copy) is
/// free to use as the poll interval.
// ponytail: a client that half-closes its write side after sending (shutdown(SHUT_WR)) reads as
// gone; browsers, curl, requests and fetch do not. Poll interval = cancel latency.
fn watch_for_close(stream: &TcpStream, flag: &AtomicBool, done: &AtomicBool) {
    stream.set_read_timeout(Some(WATCH_INTERVAL)).ok();
    let mut one = [0u8; 1];
    while !done.load(Ordering::Relaxed) {
        match stream.peek(&mut one) {
            Ok(0) => return flag.store(true, Ordering::Relaxed),
            // pipelined bytes are waiting: not a close, and peek would return at once forever
            Ok(_) => std::thread::sleep(WATCH_INTERVAL),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return flag.store(true, Ordering::Relaxed),
        }
    }
}

/// How often a running request checks whether its client is still there.
const WATCH_INTERVAL: Duration = Duration::from_millis(25);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(s: &str) -> Result<Request, HttpError> {
        read_request(&mut Cursor::new(s.as_bytes().to_vec()), 1024)
    }

    fn status(s: &str) -> u16 {
        parse(s).unwrap_err().status
    }

    #[test]
    fn parses_a_get_with_query_and_headers() {
        let r = parse("GET /api/x%20y?a=1&b=two+words&c HTTP/1.1\r\nHost: localhost:1\r\nX-Custom:  v \r\n\r\n").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str()), ("GET", "/api/x y"));
        assert_eq!(r.param("a"), Some("1"));
        assert_eq!(r.param("b"), Some("two words"));
        assert_eq!(r.param("c"), Some(""));
        assert_eq!(r.param("z"), None);
        assert_eq!(r.header("x-custom"), Some("v"));
        assert_eq!(r.header("host"), Some("localhost:1"));
        assert!(r.body.is_empty());
    }

    #[test]
    fn reads_exactly_content_length_bytes() {
        let r = parse("POST /p HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloEXTRA").unwrap();
        assert_eq!(r.body, b"hello");
        let r = parse("PUT /p HTTP/1.0\r\nContent-Length: 0\r\n\r\n").unwrap();
        assert!(r.body.is_empty());
    }

    #[test]
    fn hostile_requests_get_the_right_status() {
        assert_eq!(status(""), 400);
        assert_eq!(status("GET / HTTP/1.1\r\n"), 400);
        assert_eq!(status("GARBAGE\r\n\r\n"), 400);
        assert_eq!(status("get / HTTP/1.1\r\n\r\n"), 400);
        assert_eq!(status("GET / HTTP/2\r\n\r\n"), 400);
        assert_eq!(status("GET / HTTP/1.1 extra\r\n\r\n"), 400);
        assert_eq!(status("GET nothing HTTP/1.1\r\n\r\n"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nbadheader\r\n\r\n"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nbad name: x\r\n\r\n"), 400);
        assert_eq!(status("GET /%zz HTTP/1.1\r\n\r\n"), 400);
        assert_eq!(status("GET /%ff HTTP/1.1\r\n\r\n"), 400);
        assert_eq!(status("POST / HTTP/1.1\r\n\r\n"), 411);
        assert_eq!(status("POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n"), 400);
        assert_eq!(
            status("POST / HTTP/1.1\r\nContent-Length: 2000\r\n\r\n"),
            413
        );
        assert_eq!(
            status("POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc"),
            400
        );
        assert_eq!(
            status("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"),
            501
        );
        let long = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert_eq!(status(&long), 431);
        let many: String = (0..=MAX_HEADERS).map(|i| format!("H{i}: v\r\n")).collect();
        assert_eq!(status(&format!("GET / HTTP/1.1\r\n{many}\r\n")), 431);
    }

    #[test]
    fn percent_decoding_handles_edges() {
        assert_eq!(percent_decode("a%2Fb", false).unwrap(), "a/b");
        assert_eq!(percent_decode("a+b", false).unwrap(), "a+b");
        assert_eq!(percent_decode("a+b", true).unwrap(), "a b");
        assert!(percent_decode("%4", false).is_err());
        assert!(percent_decode("%", false).is_err());
        assert_eq!(percent_decode("", false).unwrap(), "");
    }

    #[test]
    fn a_stalled_client_times_out_with_408() {
        struct Stall;
        impl Read for Stall {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::WouldBlock.into())
            }
        }
        assert_eq!(read_request(&mut Stall, 10).unwrap_err().status, 408);
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::ConnectionReset.into())
            }
        }
        assert_eq!(read_request(&mut Broken, 10).unwrap_err().status, 400);
        struct StallBody(Vec<u8>);
        impl Read for StallBody {
            fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
                if self.0.is_empty() {
                    return Err(std::io::ErrorKind::TimedOut.into());
                }
                b[0] = self.0.remove(0);
                Ok(1)
            }
        }
        let head = b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\nab".to_vec();
        assert_eq!(
            read_request(&mut StallBody(head), 10).unwrap_err().status,
            408
        );
    }

    #[test]
    fn responses_serialise_with_length_and_safe_headers() {
        let mut out = Vec::new();
        let mut r = Response::new(200, "text/plain", "hi");
        r.headers.push(("X-A".into(), "b".into()));
        write_response(&mut out, &r).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(s.contains("Content-Length: 2\r\n") && s.contains("X-A: b\r\n"));
        assert!(
            s.contains("X-Content-Type-Options: nosniff") && s.contains("Cache-Control: no-store")
        );
        assert!(s.ends_with("\r\n\r\nhi"));
        for (code, text) in [
            (204, "No Content"),
            (400, "Bad Request"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (405, "Method Not Allowed"),
            (408, "Request Timeout"),
            (409, "Conflict"),
            (411, "Length Required"),
            (413, "Payload Too Large"),
            (431, "Request Header Fields Too Large"),
            (500, "Internal Server Error"),
            (501, "Not Implemented"),
            (401, "Unauthorized"),
            (422, "Unprocessable Entity"),
            (503, "Service Unavailable"),
            (999, "Unknown"),
        ] {
            assert_eq!(reason(code), text);
        }
        let e = Response::error(404, "nope");
        assert_eq!(e.status, 404);
        assert_eq!(String::from_utf8(e.body).unwrap(), "{\"error\":\"nope\"}");
    }

    #[test]
    fn the_guard_accepts_only_this_loopback_server() {
        let req = |h: &[(&str, &str)]| Request {
            method: "GET".into(),
            path: "/".into(),
            query: vec![],
            headers: h
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            body: vec![],
        };
        assert!(guard(&req(&[("host", "127.0.0.1:80")]), 80).is_none());
        assert!(guard(&req(&[("host", "localhost:80")]), 80).is_none());
        assert!(guard(
            &req(&[("host", "localhost:80"), ("origin", "http://localhost:80")]),
            80
        )
        .is_none());
        assert_eq!(guard(&req(&[]), 80).unwrap().status, 403);
        assert_eq!(
            guard(&req(&[("host", "evil.example:80")]), 80)
                .unwrap()
                .status,
            403
        );
        assert_eq!(
            guard(&req(&[("host", "127.0.0.1:81")]), 80).unwrap().status,
            403
        );
        let cross = guard(
            &req(&[("host", "127.0.0.1:80"), ("origin", "http://evil.example")]),
            80,
        );
        assert_eq!(cross.unwrap().status, 403);
    }

    #[test]
    fn the_server_answers_over_tcp_applies_the_guard_and_stops() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let handler: Handler = Arc::new(|r: &Request| {
            Response::new(200, "text/plain", format!("{} {}", r.method, r.path))
        });
        let (s2, h2) = (stop.clone(), handler.clone());
        let t = std::thread::spawn(move || serve(listener, h2, s2, 1024));
        let ask = |raw: String| {
            let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
            c.write_all(raw.as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            out
        };
        let ok = ask(format!(
            "GET /hello HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n"
        ));
        assert!(
            ok.starts_with("HTTP/1.1 200") && ok.ends_with("GET /hello"),
            "{ok}"
        );
        let denied = ask("GET /hello HTTP/1.1\r\nHost: evil.example\r\n\r\n".into());
        assert!(denied.starts_with("HTTP/1.1 403"), "{denied}");
        let malformed = ask("nonsense\r\n\r\n".into());
        assert!(malformed.starts_with("HTTP/1.1 400"), "{malformed}");
        stop.store(true, Ordering::Relaxed);
        t.join().unwrap();
    }

    /// Serve `work`, which runs on the request thread; returns the port and a stop switch.
    fn serve_work(
        work: impl Fn(&Request) -> Response + Send + Sync + 'static,
    ) -> (u16, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let (s, h): (_, Handler) = (stop.clone(), Arc::new(work));
        let t = std::thread::spawn(move || serve_with(listener, h, s, 1024, false));
        (port, stop, t)
    }

    #[test]
    fn a_client_that_disconnects_mid_request_cancels_the_work() {
        let (tx, rx) = std::sync::mpsc::channel();
        let tx = std::sync::Mutex::new(tx);
        let (port, stop, t) = serve_work(move |_| {
            // stands in for a token loop: runs until cancelled, at most 10 s
            let t0 = std::time::Instant::now();
            while crate::cancel::check().is_ok() && t0.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
            }
            tx.lock().unwrap().send(t0.elapsed()).unwrap();
            Response::new(200, "text/plain", "late")
        });
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        c.write_all(b"POST /work HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        std::thread::sleep(Duration::from_millis(150)); // the work is running
        drop(c); // the client goes away
        let took = rx.recv_timeout(Duration::from_secs(5)).expect("cancelled");
        assert!(took < Duration::from_secs(3), "stopped promptly: {took:?}");
        stop.store(true, Ordering::Relaxed);
        t.join().unwrap();
    }

    #[test]
    fn a_client_that_stays_is_never_cancelled_and_gets_its_answer() {
        let (port, stop, t) = serve_work(|_| {
            // slow enough for several watcher polls
            std::thread::sleep(Duration::from_millis(200));
            match crate::cancel::check() {
                Ok(()) => Response::new(200, "text/plain", "done"),
                Err(e) => Response::new(500, "text/plain", e),
            }
        });
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        c.write_all(b"POST /work HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        assert!(
            out.starts_with("HTTP/1.1 200") && out.ends_with("done"),
            "{out}"
        );
        // a GET is not watched at all: the flag is never installed
        let mut g = TcpStream::connect(("127.0.0.1", port)).unwrap();
        g.write_all(b"GET /w HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut out = String::new();
        g.read_to_string(&mut out).unwrap();
        assert!(out.ends_with("done"), "{out}");
        stop.store(true, Ordering::Relaxed);
        t.join().unwrap();
    }
}
