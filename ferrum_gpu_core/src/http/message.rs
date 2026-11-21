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
