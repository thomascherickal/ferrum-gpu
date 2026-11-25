use super::head::{parse_head, read_head};
use super::*;

/// Read one request from `r`, refusing heads over [`MAX_HEAD`] and bodies over `max_body`, within
/// the default [`Budget`].
pub fn read_request(r: &mut dyn Read, max_body: usize) -> Result<Request, HttpError> {
    read_request_within(r, max_body, Budget::default(), &Instant::now)
}

/// [`read_request`] with a total time `budget` on `now`'s clock: 408 once the head takes longer
/// than `budget.head`, or the body longer than `budget.body(len)` after the head.
pub fn read_request_within(
    r: &mut dyn Read,
    max_body: usize,
    budget: Budget,
    now: &dyn Fn() -> Instant,
) -> Result<Request, HttpError> {
    let start = now();
    let mut timed = Timed {
        inner: r,
        now,
        start,
        allowed: budget.head,
    };
    let (method, path, query, headers) = parse_head(&read_head(&mut timed)?)?;
    let find = |n: &str| headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    if find("transfer-encoding").is_some() {
        return Err(bad(
            501,
            "chunked or encoded request bodies are not supported",
        ));
    }
    let lengths = headers.iter().filter(|(k, _)| k == "content-length");
    if lengths
        .clone()
        .any(|(_, v)| Some(v) != find("content-length").as_ref())
    {
        return Err(bad(400, "conflicting Content-Length headers"));
    }
    let len = match find("content-length") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| bad(400, "bad Content-Length"))?,
        None if matches!(method.as_str(), "POST" | "PUT") => {
            return Err(bad(411, "Content-Length required"))
        }
        None => 0,
    };
    if len > max_body {
        return Err(bad(413, "request body too large"));
    }
    let mut body = vec![0u8; len];
    (timed.start, timed.allowed) = (now(), budget.body(len));
    timed.read_exact(&mut body).map_err(|e| {
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
        method,
        path,
        query,
        headers,
        body,
    })
}
