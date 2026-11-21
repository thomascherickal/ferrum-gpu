use super::*;

/// Read one request from `r`, refusing heads over [`MAX_HEAD`] and bodies over `max_body`.
pub fn read_request(r: &mut impl Read, max_body: usize) -> Result<Request, HttpError> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match r.read(&mut byte) {
            Ok(0) => return Err(bad(400, "connection closed before the request head ended")),
            Ok(_) => head.push(byte[0]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return Err(bad(408, "timed out reading the request"))
            }
            Err(_) => return Err(bad(400, "read error")),
        }
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
        if head.len() > MAX_HEAD {
            return Err(bad(431, "request head too large"));
        }
    }
    let text = std::str::from_utf8(&head).map_err(|_| bad(400, "request head is not UTF-8"))?;
    let mut lines = text.split("\r\n");
    let line = lines.next().unwrap_or("");
    let mut parts = line.split(' ');
    let (method, target, version) = (
        parts.next().unwrap_or(""),
        parts.next().unwrap_or(""),
        parts.next().unwrap_or(""),
    );
    if method.is_empty()
        || target.is_empty()
        || parts.next().is_some()
        || !version.starts_with("HTTP/1.")
    {
        return Err(bad(400, "malformed request line"));
    }
    if !method.bytes().all(|c| c.is_ascii_uppercase()) {
        return Err(bad(400, "malformed method"));
    }
    let mut headers: Vec<(String, String)> = Vec::new();
    for l in lines.filter(|l| !l.is_empty()) {
        let (k, v) = l
            .split_once(':')
            .ok_or_else(|| bad(400, "malformed header"))?;
        if k.is_empty() || k.contains(' ') {
            return Err(bad(400, "malformed header name"));
        }
        headers.push((k.to_ascii_lowercase(), v.trim().to_string()));
        if headers.len() > MAX_HEADERS {
            return Err(bad(431, "too many headers"));
        }
    }
    let (raw_path, raw_query) = target.split_once('?').unwrap_or((target, ""));
    if !raw_path.starts_with('/') {
        return Err(bad(400, "request target must be an absolute path"));
    }
    let path = percent_decode(raw_path, false)?;
    let mut query = Vec::new();
    for kv in raw_query.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        query.push((percent_decode(k, true)?, percent_decode(v, true)?));
    }
    let find = |n: &str| headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    if find("transfer-encoding").is_some() {
        return Err(bad(
            501,
            "chunked or encoded request bodies are not supported",
        ));
    }
    let len = match find("content-length") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| bad(400, "bad Content-Length"))?,
        None if matches!(method, "POST" | "PUT") => {
            return Err(bad(411, "Content-Length required"))
        }
        None => 0,
    };
    if len > max_body {
        return Err(bad(413, "request body too large"));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).map_err(|e| {
        if matches!(
            e.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ) {
            bad(408, "timed out reading the body")
        } else {
            bad(400, "body shorter than Content-Length")
        }
    })?;
    Ok(Request {
        method: method.to_string(),
        path,
        query,
        headers,
        body,
    })
}

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
