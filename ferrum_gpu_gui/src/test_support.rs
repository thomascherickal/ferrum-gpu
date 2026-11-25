//! Request and reply helpers shared by the route tests.
use crate::http::{Request, Response};
use crate::json::Json;

/// A `method` request for `path` with `query` pairs and `body`.
pub(crate) fn request(method: &str, path: &str, query: &[(&str, &str)], body: &[u8]) -> Request {
    Request {
        method: method.into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: body.to_vec(),
    }
}

/// The status of `r` and its body as JSON (`Null` when it is not JSON).
pub(crate) fn reply(r: Response) -> (u16, Json) {
    let body = Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null);
    (r.status, body)
}

/// Field `k` of `j`, `Null` when absent.
pub(crate) fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}
