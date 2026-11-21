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
    let r = parse(
        "GET /api/x%20y?a=1&b=two+words&c HTTP/1.1\r\nHost: localhost:1\r\nX-Custom:  v \r\n\r\n",
    )
    .unwrap();
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
    assert!(s.contains("X-Content-Type-Options: nosniff") && s.contains("Cache-Control: no-store"));
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
        assert!(
            crate::cancel::check().is_ok(),
            "a client that stays is not cancelled"
        );
        Response::new(200, "text/plain", "done")
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

#[test]
fn extra_bytes_after_a_request_are_not_mistaken_for_a_disconnect() {
    // a client that pipelines a second request leaves bytes waiting on the socket; peeking
    // returns them at once, forever, and they say nothing about the client leaving
    let (port, stop, t) = serve_work(|_| {
        std::thread::sleep(Duration::from_millis(200)); // several watcher polls
        assert!(
            crate::cancel::check().is_ok(),
            "a client that stays is not cancelled"
        );
        Response::new(200, "text/plain", "done")
    });
    let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
    c.write_all(
        b"POST /a HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\nGET /b HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    assert!(
        out.starts_with("HTTP/1.1 200") && out.ends_with("done"),
        "{out}"
    );
    stop.store(true, Ordering::Relaxed);
    t.join().unwrap();
}

#[test]
fn the_close_watcher_tells_a_close_a_reset_and_silence_apart() {
    use std::io::ErrorKind;
    let run = |script: Vec<std::io::Result<usize>>| {
        let flag = AtomicBool::new(false);
        let done = AtomicBool::new(false);
        let mut script = script.into_iter();
        watch_with(|| script.next().unwrap_or(Ok(0)), &flag, &done);
        flag.load(Ordering::Relaxed)
    };
    // nothing yet, then pipelined bytes (neither is a close), then the peer closes
    assert!(run(vec![
        Err(ErrorKind::WouldBlock.into()),
        Err(ErrorKind::TimedOut.into()),
        Ok(3),
        Ok(0)
    ]));
    // a reset is a close too
    assert!(run(vec![Err(ErrorKind::ConnectionReset.into())]));
    // the handler finishing first leaves the flag alone
    let flag = AtomicBool::new(false);
    let done = AtomicBool::new(true);
    watch_with(|| panic!("never polled"), &flag, &done);
    assert!(!flag.load(Ordering::Relaxed));
}
