use super::*;

/// Bodies of `POST /v1/generate` that must each be answered 400 with an `error`.
const BAD_GENERATE_BODIES: &[&str] = &[
    "not json",
    r#"{"prompt":"x"}"#,
    r#"{"requests":{"prompt":"x"}}"#,
    r#"{"requests":[{"max_tokens":3}]}"#,
    r#"{"requests":[{"prompt":"x","max_tokens":0}]}"#,
    r#"{"requests":[{"prompt":"x","temperature":-1}]}"#,
    r#"{"requests":[{"prompt":"x","temperature":"hot"}]}"#,
    r#"{"requests":[{"prompt":"x","top_p":0}]}"#,
    r#"{"requests":[{"prompt":"x","top_p":1.5}]}"#,
    r#"{"requests":[{"prompt":"x","repetition_penalty":0}]}"#,
    r#"{"requests":[{"prompt":"x","top_k":-1}]}"#,
    r#"{"requests":[{"prompt":"x","chat":true}]}"#,
    r#"{"requests":[{"prompt":"x","stop":"a"}]}"#,
    r#"{"requests":[{"prompt":"x","stop":[1]}]}"#,
    r#"{"requests":[{"prompt":"x","stop":[""]}]}"#,
    r#"{"requests":[{"prompt":"x","bogus":1}]}"#,
    r#"{"requests":[],"extra":1}"#,
    "[]",
];

/// Other bad calls: method, path, body, the status they must get.
const OTHER_BAD_CALLS: &[(&str, &str, &str, u16)] = &[
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
];

#[test]
fn bad_requests_get_clear_errors() {
    let generate = BAD_GENERATE_BODIES
        .iter()
        .map(|b| ("POST", "/v1/generate", *b, 400));
    for (m, p, b, status) in generate.chain(OTHER_BAD_CALLS.iter().copied()) {
        let (s, j) = call(m, p, b);
        assert_eq!(s, status, "{m} {p} {b}");
        assert!(j.get("error").is_some(), "{m} {p} {b}");
    }
    let stops = (0..17)
        .map(|i| format!("\"s{i}\""))
        .collect::<Vec<_>>()
        .join(",");
    let many = format!(r#"{{"requests":[{{"prompt":"x","stop":[{stops}]}}]}}"#);
    assert_eq!(call("POST", "/v1/generate", &many).0, 400);
    let long = "s".repeat(MAX_STOP_BYTES + 1);
    let long_stop = format!(r#"{{"requests":[{{"prompt":"x","stop":["{long}"]}}]}}"#);
    assert_eq!(call("POST", "/v1/generate", &long_stop).0, 400);
    let mut r = req("POST", "/v1/generate", "");
    r.body = vec![0xFF, 0xFE];
    assert_eq!(route(&StubDecoder, &Limits::default(), &r).status, 400);
}
