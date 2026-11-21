use super::*;

/// Like [`serve`]; with `loopback_guard` false the Host/Origin check is skipped, for a server
/// that listens beyond loopback and authenticates every request itself.
pub fn serve_with(
    listener: TcpListener,
    handler: Handler,
    stop: Arc<AtomicBool>,
    max_body: usize,
    loopback_guard: bool,
) {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    listener.set_nonblocking(true).ok();
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let h = handler.clone();
                std::thread::spawn(move || handle(stream, &*h, (port, loopback_guard), max_body));
            }
            // no connection waiting (WouldBlock) or a transient accept error: look again shortly
            Err(_) => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

pub(super) fn handle(
    mut stream: TcpStream,
    handler: &(dyn Fn(&Request) -> Response + Send + Sync),
    (port, loopback_guard): (u16, bool),
    max_body: usize,
) {
    stream.set_nonblocking(false).ok();
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
    let resp = match read_request(&mut stream, max_body) {
        Ok(req) => loopback_guard
            .then(|| guard(&req, port))
            .flatten()
            .unwrap_or_else(|| handle_watched(&stream, &req, handler)),
        Err(e) => Response::error(e.status, &e.message),
    };
    let _ = write_response(&mut stream, &resp);
    close_gracefully(&mut stream);
}

/// Close after the response without losing it. Closing a socket that still has unread bytes in its
/// receive buffer (a client that pipelined a second request, or sent a body we refused) makes the
/// kernel send a reset, which can destroy the response before the client has read it. So: finish
/// our side (`FIN`), swallow whatever is still arriving for a moment, then close.
pub(super) fn close_gracefully(stream: &mut TcpStream) {
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Write);
    let _ = stream.set_read_timeout(Some(DRAIN_TIMEOUT));
    let (mut sink, mut taken) = ([0u8; 4096], 0usize);
    while taken < DRAIN_LIMIT {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(n) => taken += n,
        }
    }
}

/// How long to wait for a client's last bytes after the response (a client that has read the
/// response and closed ends this at once).
pub(super) const DRAIN_TIMEOUT: Duration = Duration::from_millis(100);
/// Most bytes swallowed after the response; a client sending more is cut off by the close.
pub(super) const DRAIN_LIMIT: usize = 1 << 20;

/// Run `handler` for `req`; for a POST (the requests that compute), raise this thread's cancel
/// flag ([`crate::cancel`]) as soon as the client closes its end of `stream`.
pub(super) fn handle_watched(
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
pub(super) fn watch_for_close(stream: &TcpStream, flag: &AtomicBool, done: &AtomicBool) {
    stream.set_read_timeout(Some(WATCH_INTERVAL)).ok();
    let mut one = [0u8; 1];
    watch_with(|| stream.peek(&mut one), flag, done);
}

/// [`watch_for_close`] over any `peek`: 0 bytes is a close, an error other than "nothing yet" is a
/// reset, bytes waiting are neither.
pub(super) fn watch_with(
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
pub(super) const WATCH_INTERVAL: Duration = Duration::from_millis(25);
