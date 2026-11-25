use super::*;
use ferrum_test_support::load::{ask, drive};
use std::time::{Duration, Instant};

/// Two slow generations hold both slots; then 24 concurrent POSTs must all get the documented 503
/// at once (no hang, no crash), a GET still answers, and the slots come back when the holders leave.
#[test]
fn past_the_in_flight_cap_every_post_is_refused_at_once() {
    let d = Arc::new(SlowDecoder {
        tokens: 0.into(),
        cancelled: AtomicBool::new(false),
        entered: 0.into(),
    });
    let cfg = Config {
        max_inflight: 2,
        ..Config::default()
    };
    let r = start(d.clone(), &cfg).unwrap();
    let addr = r.addr;
    let post = |path: &str, body: &str| {
        format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
    };
    let gen = post("/v1/generate", r#"{"requests":[{"prompt":"x"}]}"#);
    let holders: Vec<TcpStream> = (0..2)
        .map(|_| {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(gen.as_bytes()).unwrap();
            s
        })
        .collect();
    let t0 = Instant::now();
    while d.entered.load(Ordering::SeqCst) < 2 {
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "holders never started"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let (answers, _) = drive(addr, 8, 3, |_| gen.clone().into_bytes());
    assert_eq!(answers.len(), 24);
    for a in &answers {
        assert_eq!(a.status, 503, "{}", a.body);
        assert_eq!(a.body, r#"{"error":"too many requests in flight"}"#);
        assert!(a.took < Duration::from_secs(5), "{:?}", a.took);
    }
    let health = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    assert_eq!(ask(addr, health.as_bytes()).status, 200);
    drop(holders); // the holders leave: their generations are cancelled and the slots freed
    let score = post(
        "/v1/loglikelihood",
        r#"{"requests":[{"context":"a","continuation":"b"}]}"#,
    );
    let t0 = Instant::now();
    while ask(addr, score.as_bytes()).status != 200 {
        assert!(t0.elapsed() < Duration::from_secs(10), "slots never freed");
        std::thread::sleep(Duration::from_millis(20));
    }
    r.stop();
}
