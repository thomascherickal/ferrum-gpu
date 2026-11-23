use super::*;

type Pairs = Vec<(String, String)>;

/// Read bytes up to and including the blank line that ends the head.
pub(super) fn read_head(r: &mut dyn Read) -> Result<Vec<u8>, HttpError> {
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
    Ok(head)
}

/// The method, decoded path, decoded query pairs and lower-cased headers of a request head.
pub(super) fn parse_head(head: &[u8]) -> Result<(String, String, Pairs, Pairs), HttpError> {
    let text = std::str::from_utf8(head).map_err(|_| bad(400, "request head is not UTF-8"))?;
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
    Ok((method.to_string(), path, query, headers))
}
