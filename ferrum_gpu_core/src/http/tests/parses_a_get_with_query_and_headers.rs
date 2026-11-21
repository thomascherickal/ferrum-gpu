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
