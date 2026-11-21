use super::*;

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
