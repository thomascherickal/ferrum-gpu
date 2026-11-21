use super::*;

/// Serialise `resp` onto `w`.
pub fn write_response(w: &mut impl Write, resp: &Response) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n",
        resp.status,
        reason(resp.status),
        resp.content_type,
        resp.body.len()
    );
    for (k, v) in &resp.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(&resp.body)?;
    w.flush()
}

/// `Some(response)` when `req` must be refused: its `Host` is not this loopback server, or it
/// carries a cross-site `Origin`.
pub fn guard(req: &Request, port: u16) -> Option<Response> {
    let ok_hosts = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    match req.header("host") {
        Some(h) if ok_hosts.iter().any(|o| o == h) => {}
        _ => return Some(Response::error(403, "unexpected Host header")),
    }
    if let Some(o) = req.header("origin") {
        if !ok_hosts.iter().any(|h| o == format!("http://{h}")) {
            return Some(Response::error(403, "cross-origin request refused"));
        }
    }
    None
}

/// The handler type the server calls for every accepted request.
pub type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

/// Serve `listener` until `stop` is set: one thread per connection, loopback guard applied
/// before `handler` runs. Returns when the stop flag is seen (the flag is polled between accepts).
pub fn serve(listener: TcpListener, handler: Handler, stop: Arc<AtomicBool>, max_body: usize) {
    serve_with(listener, handler, stop, max_body, true);
}
