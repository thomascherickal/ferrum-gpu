use super::*;

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
