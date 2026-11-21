use super::*;

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
    // ferrum-cpu's LightEval adapter reads `context_len`
    assert_eq!(j.get("context_len").and_then(Json::as_usize), Some(16));
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
