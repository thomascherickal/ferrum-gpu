//! A load test of the connection cap: the cap's worth of clients hold their connections, a crowd
//! past the cap gets a quick 503 with `Retry-After`, and once the holders finish the full capacity
//! serves again. The `LOAD` line goes to system-profile.md (`scripts/load_profile.py`).
use super::*;
use ferrum_test_support::load::{ask, drive, summary};
use std::sync::atomic::AtomicUsize;

const CAP: usize = 4;

#[test]
fn past_the_connection_cap_clients_are_refused_and_capacity_returns() {
    let (entered, release) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicBool::new(false)),
    );
    let (e, r) = (entered.clone(), release.clone());
    let mut opts = ServeOptions::new(1024, false);
    opts.max_connections = CAP;
    let (port, stop, t) = serve_work_with(opts, move |_| {
        e.fetch_add(1, Ordering::SeqCst);
        while !r.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        Response::new(200, "text/plain", "ok")
    });
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let get = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
    let holders: Vec<_> = (0..CAP)
        .map(|_| std::thread::spawn(move || ask(addr, get).status))
        .collect();
    while entered.load(Ordering::SeqCst) < CAP {
        std::thread::sleep(Duration::from_millis(5));
    }
    let (refused, wall) = drive(addr, 16, 2, |_| get.to_vec());
    println!("{}", summary("connection-cap", 16, &refused, wall));
    for a in &refused {
        assert_eq!(
            (a.status, a.body.as_str()),
            (503, r#"{"error":"too many connections"}"#)
        );
        assert!(
            a.took < Duration::from_secs(1),
            "refused quickly: {:?}",
            a.took
        );
    }
    let mut raw = String::new();
    let mut c = TcpStream::connect(addr).unwrap();
    c.write_all(get).unwrap();
    c.read_to_string(&mut raw).unwrap();
    assert!(raw.contains("\r\nRetry-After: 1\r\n"), "{raw}");
    release.store(true, Ordering::SeqCst);
    for h in holders {
        assert_eq!(h.join().unwrap(), 200);
    }
    // the holders' places come back; give their threads a moment to end
    std::thread::sleep(Duration::from_millis(300));
    let (served, wall) = drive(addr, CAP, 8, |_| get.to_vec());
    println!("{}", summary("connection-cap-after", CAP, &served, wall));
    assert!(served.iter().all(|a| a.status == 200));
    stop.store(true, Ordering::Relaxed);
    t.join().unwrap();
}
