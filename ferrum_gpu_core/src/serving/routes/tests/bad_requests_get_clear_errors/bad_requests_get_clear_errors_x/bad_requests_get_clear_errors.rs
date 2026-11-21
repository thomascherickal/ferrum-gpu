use super::*;

#[test]
pub(super) fn bad_requests_get_clear_errors() {
    for (m, p, b, status) in [
        ("POST", "/v1/generate", "not json", 400),
        ("POST", "/v1/generate", r#"{"prompt":"x"}"#, 400),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":{"prompt":"x"}}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"max_tokens":3}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","max_tokens":0}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","temperature":-1}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","temperature":"hot"}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","top_p":0}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","top_p":1.5}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","repetition_penalty":0}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","top_k":-1}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","chat":true}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","stop":"a"}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","stop":[1]}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","stop":[""]}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"x","bogus":1}]}"#,
            400,
        ),
        ("POST", "/v1/generate", r#"{"requests":[],"extra":1}"#, 400),
        ("POST", "/v1/generate", "[]", 400),
        (
            "POST",
            "/v1/loglikelihood",
            r#"{"requests":[{"context":"a"}]}"#,
            400,
        ),
        (
            "POST",
            "/v1/loglikelihood",
            r#"{"requests":[{"context":"a","continuation":"b","x":1}]}"#,
            400,
        ),
        ("POST", "/v1/loglikelihood_rolling", r#"{"texts":[1]}"#, 400),
        ("POST", "/v1/loglikelihood_rolling", r#"{"text":"a"}"#, 400),
        ("GET", "/v1/generate", "", 405),
        ("GET", "/v1/loglikelihood_rolling", "", 405),
        ("POST", "/health", "", 405),
        ("GET", "/nope", "", 404),
    ] {
        let (s, j) = call(m, p, b);
        assert_eq!(s, status, "{m} {p} {b}");
        assert!(j.get("error").is_some(), "{m} {p} {b}");
    }
    let many = format!(
        r#"{{"requests":[{{"prompt":"x","stop":[{}]}}]}}"#,
        (0..17)
            .map(|i| format!("\"s{i}\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    assert_eq!(call("POST", "/v1/generate", &many).0, 400);
    let long_stop = format!(
        r#"{{"requests":[{{"prompt":"x","stop":["{}"]}}]}}"#,
        "s".repeat(MAX_STOP_BYTES + 1)
    );
    assert_eq!(call("POST", "/v1/generate", &long_stop).0, 400);
    let mut r = req("POST", "/v1/generate", "");
    r.body = vec![0xFF, 0xFE];
    assert_eq!(route(&StubDecoder, &Limits::default(), &r).status, 400);
}
