use super::*;

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
