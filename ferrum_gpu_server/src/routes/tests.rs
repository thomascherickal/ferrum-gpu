use super::*;
use crate::decoder::StubDecoder;

fn req(method: &str, path: &str, body: &str) -> Request {
    Request {
        method: method.into(),
        path: path.into(),
        query: vec![],
        headers: vec![],
        body: body.as_bytes().to_vec(),
    }
}

fn call(method: &str, path: &str, body: &str) -> (u16, Json) {
    let r = route(&StubDecoder, &req(method, path, body));
    (
        r.status,
        Json::parse(std::str::from_utf8(&r.body).unwrap()).unwrap(),
    )
}

#[test]
fn health_and_info() {
    assert_eq!(
        call("GET", "/health", "")
            .1
            .get("status")
            .and_then(Json::as_str),
        Some("ok")
    );
    let (s, j) = call("GET", "/v1/info", "");
    assert_eq!(
        (
            s,
            j.get("device").and_then(Json::as_str),
            j.get("ctx").and_then(Json::as_usize)
        ),
        (200, Some("stub"), Some(16))
    );
}

#[test]
fn generate_uses_defaults_and_options() {
    let (s, j) = call("POST", "/v1/generate", r#"{"prompt":"hi"}"#);
    assert_eq!(s, 200);
    assert_eq!(j.get("tokens").and_then(Json::as_usize), Some(64));
    let (_, j) = call(
        "POST",
        "/v1/generate",
        r#"{"prompt":"hi","max_new":3,"temperature":0.5,"top_k":4,"seed":9}"#,
    );
    assert_eq!(j.get("text").and_then(Json::as_str), Some("!!!"));
}

#[test]
fn loglikelihood_sums_the_continuation_only() {
    // ids: "ab" + "c" = [97,98,99]; scores for 98 (odd) and 99 (odd); only the last is the continuation.
    let (s, j) = call(
        "POST",
        "/v1/loglikelihood",
        r#"{"context":"ab","continuation":"c"}"#,
    );
    assert_eq!(s, 200);
    assert_eq!(j.get("loglikelihood").and_then(Json::as_f64), Some(-1.0));
    assert_eq!(j.get("tokens").and_then(Json::as_usize), Some(1));
    assert_eq!(j.get("is_greedy"), Some(&Json::Bool(false)));
    let (_, j) = call(
        "POST",
        "/v1/loglikelihood",
        r#"{"context":"a","continuation":"bb"}"#,
    );
    assert_eq!(j.get("loglikelihood").and_then(Json::as_f64), Some(-2.0));
    let (_, j) = call(
        "POST",
        "/v1/loglikelihood",
        r#"{"context":"a","continuation":"b"}"#,
    );
    assert_eq!(j.get("is_greedy"), Some(&Json::Bool(true)));
}

#[test]
fn bad_requests_get_clear_errors() {
    for (m, p, b, status) in [
        ("POST", "/v1/generate", "not json", 400),
        ("POST", "/v1/generate", r#"{"max_new":3}"#, 400),
        ("POST", "/v1/generate", r#"{"prompt":"x","max_new":0}"#, 400),
        (
            "POST",
            "/v1/generate",
            r#"{"prompt":"x","max_new":5000}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"prompt":"x","temperature":"hot"}"#,
            400,
        ),
        (
            "POST",
            "/v1/loglikelihood",
            r#"{"context":"","continuation":"b"}"#,
            400,
        ),
        ("POST", "/v1/loglikelihood", r#"{"context":"a"}"#, 400),
        ("GET", "/v1/generate", "", 405),
        ("POST", "/health", "", 405),
        ("GET", "/nope", "", 404),
    ] {
        let (s, j) = call(m, p, b);
        assert_eq!(s, status, "{m} {p} {b}");
        assert!(j.get("error").is_some());
    }
    let long = format!(r#"{{"prompt":"{}"}}"#, "x".repeat(MAX_PROMPT + 1));
    assert_eq!(call("POST", "/v1/generate", &long).0, 400);
    let mut r = req("POST", "/v1/generate", "");
    r.body = vec![0xFF, 0xFE];
    assert_eq!(route(&StubDecoder, &r).status, 400);
}
