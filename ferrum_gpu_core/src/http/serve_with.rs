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

fn handle(
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
fn close_gracefully(stream: &mut TcpStream) {
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
const DRAIN_TIMEOUT: Duration = Duration::from_millis(100);
/// Most bytes swallowed after the response; a client sending more is cut off by the close.
const DRAIN_LIMIT: usize = 1 << 20;
