use super::*;

#[test]
fn bad_input_gets_a_clear_error_and_never_a_crash() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let err = |r: Reply| -> String { field(&r.json(), "error").as_str().unwrap_or("").to_string() };
    assert!(err(post(p, "/api/model", "{")).contains("bad JSON"));
    assert!(err(post(p, "/api/model", r#"{"preset":"huge"}"#)).contains("unknown preset"));
    assert!(err(post(p, "/api/model", r#"{"seed":-1}"#)).contains("seed"));
    assert!(err(call(p, "POST", "/api/model", &[0xff, 0xfe])).contains("UTF-8"));
    assert_eq!(post(p, "/api/model", r#"{"preset":"tiny"}"#).status, 200);
    assert!(err(post(p, "/api/train", "{")).contains("bad JSON"));
    assert!(err(post(p, "/api/generate", "{")).contains("bad JSON"));
    assert!(err(post(p, "/api/train", "{}")).contains("'text'"));
    assert!(err(post(p, "/api/train", r#"{"text":"short"}"#)).contains("needs more"));
    assert!(err(post(
        p,
        "/api/train",
        &format!(r#"{{"text":"{}","steps":0}}"#, "x".repeat(100))
    ))
    .contains("steps"));
    assert!(err(post(
        p,
        "/api/train",
        &format!(r#"{{"text":"{}","lr":"fast"}}"#, "x".repeat(100))
    ))
    .contains("lr"));
    assert_eq!(
        post(
            p,
            "/api/train",
            &format!(r#"{{"text":"{}"}}"#, "x".repeat((1 << 20) + 1))
        )
        .status,
        413
    );
    assert!(err(post(p, "/api/generate", r#"{"prompt":""}"#)).contains("empty"));
    assert!(err(post(p, "/api/generate", "{}")).contains("'prompt'"));
    assert_eq!(
        post(
            p,
            "/api/generate",
            &format!(r#"{{"prompt":"{}"}}"#, "y".repeat(5000))
        )
        .status,
        413
    );
    assert!(err(post(p, "/api/generate", r#"{"prompt":"a","max_new":0}"#)).contains("max_new"));
    assert!(err(post(
        p,
        "/api/generate",
        r#"{"prompt":"a","temperature":99}"#
    ))
    .contains("temperature"));
    assert!(err(post(
        p,
        "/api/generate",
        &format!(r#"{{"prompt":"{}"}}"#, "z".repeat(200))
    ))
    .contains("context"));
    assert!(err(get(p, "/api/model.gguf?quant=q3")).contains("unknown quant"));
    assert_eq!(
        post(p, "/api/train/stop", "{}").json().get("stopping"),
        Some(&Json::Bool(false))
    );
    assert_eq!(
        field(&get(p, "/api/train/status").json(), "state").as_str(),
        Some("idle")
    );
    srv.stop();
}
