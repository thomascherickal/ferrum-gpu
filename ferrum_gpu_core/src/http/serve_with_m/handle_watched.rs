use super::*;

/// Run `handler` for `req`; for a POST (the requests that compute), raise this thread's cancel
/// flag ([`crate::cancel`]) as soon as the client closes its end of `stream`.
pub(crate) fn handle_watched(
    stream: &TcpStream,
    req: &Request,
    handler: &(dyn Fn(&Request) -> Response + Send + Sync),
) -> Response {
    let (flag, done) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let watcher = (req.method == "POST")
        .then(|| stream.try_clone().ok())
        .flatten()
        .map(|s| {
            let (f, d) = (flag.clone(), done.clone());
            std::thread::spawn(move || watch_for_close(&s, &f, &d))
        });
    let _scope = crate::cancel::install(flag);
    let resp = handler(req);
    done.store(true, Ordering::Relaxed);
    if let Some(w) = watcher {
        let _ = w.join();
    }
    resp
}

/// Raise `flag` when the peer closes or resets `stream`; return when `done` is set. The request
/// has been read completely, so the socket's read timeout (shared with the handler's copy) is
/// free to use as the poll interval.
// ponytail: a client that half-closes its write side after sending (shutdown(SHUT_WR)) reads as
// gone; browsers, curl, requests and fetch do not. Poll interval = cancel latency.
pub(crate) fn watch_for_close(stream: &TcpStream, flag: &AtomicBool, done: &AtomicBool) {
    stream.set_read_timeout(Some(WATCH_INTERVAL)).ok();
    let mut one = [0u8; 1];
    watch_with(|| stream.peek(&mut one), flag, done);
}

/// [`watch_for_close`] over any `peek`: 0 bytes is a close, an error other than "nothing yet" is a
/// reset, bytes waiting are neither.
pub(crate) fn watch_with(
    mut peek: impl FnMut() -> std::io::Result<usize>,
    flag: &AtomicBool,
    done: &AtomicBool,
) {
    while !done.load(Ordering::Relaxed) {
        match peek() {
            Ok(0) => return flag.store(true, Ordering::Relaxed),
            // pipelined bytes are waiting: not a close, and peek would return at once forever
            Ok(_) => std::thread::sleep(WATCH_INTERVAL),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return flag.store(true, Ordering::Relaxed),
        }
    }
}

/// How often a running request checks whether its client is still there.
pub(crate) const WATCH_INTERVAL: Duration = Duration::from_millis(25);
