use super::support::{put, tmp};
use crate::fsapi::routes::{handles, route};
use crate::http::Request;
use crate::json::Json;

fn get(path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    call("GET", path, query)
}

fn call(method: &str, path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: vec![],
    };
    let r = route(&req);
    (
        r.status,
        Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null),
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

mod only_the_fs_paths_are_handled;
mod preview_returns_text_and_validates_bytes;
