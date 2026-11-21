use super::*;
use crate::serving::decoder::StubDecoder;

fn req(method: &str, path: &str, body: &str) -> Request {
    Request {
        method: method.into(),
        path: path.into(),
        query: vec![],
        headers: vec![],
        body: body.as_bytes().to_vec(),
    }
}

fn call_with(l: &Limits, method: &str, path: &str, body: &str) -> (u16, Json) {
    let r = route(&StubDecoder, l, &req(method, path, body));
    (
        r.status,
        Json::parse(std::str::from_utf8(&r.body).unwrap()).unwrap(),
    )
}

fn call(method: &str, path: &str, body: &str) -> (u16, Json) {
    call_with(&Limits::default(), method, path, body)
}

fn first(j: &Json) -> &Json {
    match j.get("results") {
        Some(Json::Arr(a)) => &a[0],
        other => panic!("no results: {other:?}"),
    }
}

mod bad_requests_get_clear_errors;
mod health_and_info;
mod model_errors_are_422;
mod rolling_scores_texts;
