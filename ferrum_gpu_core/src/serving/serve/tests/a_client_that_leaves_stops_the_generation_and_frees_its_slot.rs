use super::*;

#[test]
fn a_client_that_leaves_stops_the_generation_and_frees_its_slot() {
    let d = Arc::new(SlowDecoder {
        tokens: 0.into(),
        cancelled: AtomicBool::new(false),
        entered: 0.into(),
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

#[test]
fn a_full_server_answers_a_post_with_503_and_still_answers_a_get() {
    let cfg = Config {
        max_inflight: 0,
        ..Config::default()
    };
    let r = start(Arc::new(StubDecoder), &cfg).unwrap();
    let post = http(
        r.addr,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"x"}]}"#,
    );
    assert!(
        post.contains("503") && post.contains("too many requests in flight"),
        "{post}"
    );
    assert!(http(r.addr, "GET", "/health", "", "").contains("200"));
    r.stop();
}
