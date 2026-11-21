use super::*;

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
        .map(|_| Slot::take(&n, MAX_INFLIGHT).expect("slot"))
        .collect();
    assert!(
        Slot::take(&n, MAX_INFLIGHT).is_none(),
        "the 33rd is refused"
    );
    assert_eq!(
        n.load(Ordering::SeqCst),
        MAX_INFLIGHT,
        "a refusal gives its count back"
    );
    drop(slots);
    assert_eq!(n.load(Ordering::SeqCst), 0);
    assert!(Slot::take(&n, MAX_INFLIGHT).is_some());
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
