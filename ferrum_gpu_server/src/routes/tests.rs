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
            j.get("arch").and_then(Json::as_str),
            j.get("ctx").and_then(Json::as_usize)
        ),
        (200, Some("stub"), Some("stub"), Some(16))
    );
}

#[test]
fn generate_defaults_options_and_batches() {
    let (s, j) = call("POST", "/v1/generate", r#"{"requests":[{"prompt":"hi"}]}"#);
    assert_eq!(s, 200);
    assert_eq!(first(&j).get("n_tokens").and_then(Json::as_usize), Some(64));
    let (_, j) = call(
        "POST",
        "/v1/generate",
        r#"{"requests":[{"prompt":"hi","max_tokens":3,"temperature":0.5,"top_k":4,"top_p":0.9,"seed":9,"repetition_penalty":1.1},{"prompt":"yo","max_tokens":2,"stop":["!"]}]}"#,
    );
    let Some(Json::Arr(items)) = j.get("results") else {
        panic!()
    };
    assert_eq!(items[0].get("text").and_then(Json::as_str), Some("!!!"));
    assert_eq!(
        items[1].get("text").and_then(Json::as_str),
        Some(""),
        "cut at the stop string"
    );
    assert_eq!(items[1].get("n_tokens").and_then(Json::as_usize), Some(2));
}

#[test]
fn an_omitted_max_tokens_respects_a_small_server_limit() {
    let l = Limits {
        max_tokens: 5,
        ..Limits::default()
    };
    let (_, j) = call_with(
        &l,
        "POST",
        "/v1/generate",
        r#"{"requests":[{"prompt":"hi"}]}"#,
    );
    assert_eq!(first(&j).get("n_tokens").and_then(Json::as_usize), Some(5));
}

#[test]
fn loglikelihood_returns_logprob_tokens_and_greedy() {
    let (s, j) = call(
        "POST",
        "/v1/loglikelihood",
        r#"{"requests":[{"context":"ab","continuation":"c"},{"context":"a","continuation":"bb"},{"context":"a","continuation":"b"}]}"#,
    );
    assert_eq!(s, 200);
    let Some(Json::Arr(items)) = j.get("results") else {
        panic!()
    };
    assert_eq!(items[0].get("logprob").and_then(Json::as_f64), Some(-1.0));
    assert_eq!(items[0].get("n_tokens").and_then(Json::as_usize), Some(1));
    assert_eq!(items[0].get("greedy"), Some(&Json::Bool(false))); // 'c' = 99 is odd
    assert_eq!(items[1].get("logprob").and_then(Json::as_f64), Some(-2.0));
    assert_eq!(items[2].get("greedy"), Some(&Json::Bool(true))); // 'b' = 98 is even
}

#[test]
fn rolling_scores_texts() {
    let (s, j) = call(
        "POST",
        "/v1/loglikelihood_rolling",
        r#"{"texts":["abc","d"]}"#,
    );
    assert_eq!(s, 200);
    assert_eq!(first(&j).get("n_tokens").and_then(Json::as_usize), Some(3));
    assert_eq!(first(&j).get("logprob").and_then(Json::as_f64), Some(-3.0));
    assert!(first(&j).get("greedy").is_none());
}

#[test]
fn limits_are_checked_before_compute() {
    let l = Limits {
        max_batch: 1,
        max_chars: 4,
        ..Limits::default()
    };
    let two = r#"{"texts":["a","b"]}"#;
    let (s, j) = call_with(&l, "POST", "/v1/loglikelihood_rolling", two);
    assert_eq!(s, 400);
    assert!(j
        .get("error")
        .and_then(Json::as_str)
        .unwrap()
        .contains("max_batch 1"));
    let (s, j) = call_with(
        &l,
        "POST",
        "/v1/loglikelihood_rolling",
        r#"{"texts":["abcde"]}"#,
    );
    assert_eq!(s, 400);
    assert!(j
        .get("error")
        .and_then(Json::as_str)
        .unwrap()
        .contains("max_chars 4"));
    assert_eq!(
        call_with(
            &l,
            "POST",
            "/v1/generate",
            r#"{"requests":[{"prompt":"abcde"}]}"#
        )
        .0,
        400
    );
    let (s, j) = call_with(
        &Limits {
            max_tokens: 10,
            ..Limits::default()
        },
        "POST",
        "/v1/generate",
        r#"{"requests":[{"prompt":"x","max_tokens":11}]}"#,
    );
    assert_eq!(s, 400);
    assert!(j
        .get("error")
        .and_then(Json::as_str)
        .unwrap()
        .contains("max_tokens"));
}

#[test]
fn bad_requests_get_clear_errors() {
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

#[test]
fn model_errors_are_422() {
    struct Broken;
    impl Decoder for Broken {
        fn info(&self) -> crate::decoder::Info {
            StubDecoder.info()
        }
        fn encode(&self, t: &str) -> Result<Vec<usize>, String> {
            StubDecoder.encode(t)
        }
        fn bos(&self) -> usize {
            0
        }
        fn complete(&self, _: &str, _: &GenReq) -> Result<(String, usize), String> {
            Err("prompt too long".into())
        }
        fn score(&self, _: &[usize]) -> Result<Vec<(f32, bool)>, String> {
            Err("device lost".into())
        }
    }
    let l = Limits::default();
    for (p, b) in [
        ("/v1/generate", r#"{"requests":[{"prompt":"x"}]}"#),
        (
            "/v1/loglikelihood",
            r#"{"requests":[{"context":"a","continuation":"b"}]}"#,
        ),
        ("/v1/loglikelihood_rolling", r#"{"texts":["ab"]}"#),
    ] {
        let r = route(&Broken, &l, &req("POST", p, b));
        assert_eq!(r.status, 422, "{p}");
    }
}
