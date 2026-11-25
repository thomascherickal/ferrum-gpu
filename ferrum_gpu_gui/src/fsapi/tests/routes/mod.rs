use super::support::{put, tmp};
use crate::fsapi::routes::{handles, route};
use crate::http::Request;
use crate::json::Json;
use crate::test_support::{field, reply, request};

fn get(path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    call("GET", path, query)
}

fn call(method: &str, path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    reply(route(&request(method, path, query, &[])))
}

mod only_the_fs_paths_are_handled;
mod preview_returns_text_and_validates_bytes;
