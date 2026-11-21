use super::*;

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
