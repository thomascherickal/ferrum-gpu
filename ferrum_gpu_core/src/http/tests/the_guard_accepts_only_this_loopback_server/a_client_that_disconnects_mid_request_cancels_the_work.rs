use super::*;

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
