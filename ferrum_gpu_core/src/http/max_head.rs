/// Largest request head (request line plus headers) accepted.
pub const MAX_HEAD: usize = 16 * 1024;
/// Most headers accepted.
pub const MAX_HEADERS: usize = 64;

/// A parsed request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// `GET`, `POST`, ...
    pub method: String,
    /// Path without the query string, percent-decoded.
    pub path: String,
    /// Decoded query parameters.
    pub query: Vec<(String, String)>,
    /// Header names lower-cased.
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Request {
    /// A header value by lower-case name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// A query parameter by name.
    pub fn param(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A response to send.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// `Content-Type`.
    pub content_type: String,
    /// Extra headers.
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Response {
    /// A response with the given type and body.
    pub fn new(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> Response {
        Response {
            status,
            content_type: content_type.to_string(),
            headers: Vec::new(),
            body: body.into(),
        }
    }

    /// A JSON response.
    pub fn json(status: u16, body: &crate::json::Json) -> Response {
        Response::new(status, "application/json", body.to_string())
    }

    /// A JSON error `{"error": message}`.
    pub fn error(status: u16, message: &str) -> Response {
        Response::json(
            status,
            &crate::json::Json::obj([("error", crate::json::Json::Str(message.into()))]),
        )
    }
}

/// A request that cannot be served, with the status to answer.
#[derive(Debug, PartialEq)]
pub struct HttpError {
    /// Status code.
    pub status: u16,
    /// Why.
    pub message: String,
}

pub(super) fn bad(status: u16, message: &str) -> HttpError {
    HttpError {
        status,
        message: message.to_string(),
    }
}

/// Reason phrase for the statuses this server emits.
pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

/// Percent-decode `s` (`+` is a space when `plus`); invalid escapes and non-UTF-8 are errors.
pub fn percent_decode(s: &str, plus: bool) -> Result<String, HttpError> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let h = b
                    .get(i + 1..i + 3)
                    .and_then(|h| std::str::from_utf8(h).ok())
                    .and_then(|h| u8::from_str_radix(h, 16).ok())
                    .ok_or_else(|| bad(400, "bad percent escape"))?;
                out.push(h);
                i += 3;
            }
            b'+' if plus => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| bad(400, "request target is not UTF-8"))
}
