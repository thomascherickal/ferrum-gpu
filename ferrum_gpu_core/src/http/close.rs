use super::*;

/// Answer a connection past the cap without reading it. ponytail: runs on the accept loop, at most
/// [`DRAIN_TIMEOUT`] per refusal; a flood past the cap is slowed down, not served.
pub(super) fn refuse(mut stream: TcpStream) {
    stream.set_nonblocking(false).ok();
    stream.set_write_timeout(Some(DRAIN_TIMEOUT)).ok();
    let mut busy = Response::error(503, "too many connections");
    busy.headers.push(("Retry-After".into(), "1".into()));
    let _ = write_response(&mut stream, &busy);
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
    let (mut sink, mut taken, t0) = ([0u8; 4096], 0usize, Instant::now());
    while taken < DRAIN_LIMIT && t0.elapsed() < DRAIN_TIMEOUT {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(n) => taken += n,
        }
    }
}

/// How long, in all, to wait for a client's last bytes after the response (a client that has read the
/// response and closed ends this at once).
const DRAIN_TIMEOUT: Duration = Duration::from_millis(100);
/// Most bytes swallowed after the response; a client sending more is cut off by the close.
const DRAIN_LIMIT: usize = 1 << 20;
