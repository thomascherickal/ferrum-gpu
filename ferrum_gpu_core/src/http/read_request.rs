use super::head::{parse_head, read_head};
use super::*;

/// Read one request from `r`, refusing heads over [`MAX_HEAD`] and bodies over `max_body`.
pub fn read_request(r: &mut dyn Read, max_body: usize) -> Result<Request, HttpError> {
    let (method, path, query, headers) = parse_head(&read_head(r)?)?;
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
        None if matches!(method.as_str(), "POST" | "PUT") => {
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
        method,
        path,
        query,
        headers,
        body,
    })
}
