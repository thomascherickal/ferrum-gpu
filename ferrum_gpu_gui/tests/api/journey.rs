use super::helpers::{field, losses, wait_until_done};
use super::http::{get, post};
use ferrum_gpu_gui::json::Json;

/// The whole user journey, identical for every backend.
pub(super) fn journey(port: u16) {
    assert_eq!(
        get(port, "/api/info").json().get("model"),
        Some(&Json::Null),
        "no model at first"
    );
    assert_eq!(
        post(port, "/api/generate", r#"{"prompt":"hi"}"#).status,
        400,
        "generate needs a model"
    );
    assert_eq!(
        post(port, "/api/train", r#"{"text":"abc"}"#).status,
        400,
        "train needs a model"
    );
    let created = post(port, "/api/model", r#"{"preset":"tiny","seed":7}"#);
    assert_eq!(created.status, 200, "{}", created.text());
    assert_eq!(field(&created.json(), "preset").as_str(), Some("tiny"));
    let text = "the quick brown fox jumps over the lazy dog. ".repeat(14);
    let start = post(
        port,
        "/api/train",
        &Json::obj([
            ("text", Json::Str(text)),
            ("steps", Json::Num(60.0)),
            ("lr", Json::Num(0.01)),
            ("batch", Json::Num(4.0)),
            ("context", Json::Num(24.0)),
        ])
        .to_string(),
    );
    assert_eq!(start.status, 200, "{}", start.text());
    let busy = post(port, "/api/train", r#"{"text":"another run"}"#);
    let status_now = get(port, "/api/train/status").json();
    if field(&status_now, "state").as_str() == Some("running") {
        assert_eq!(
            busy.status, 409,
            "a second run is refused while one is going"
        );
        assert_eq!(
            post(port, "/api/model", "{}").status,
            409,
            "so is replacing the model"
        );
    }
    let done = wait_until_done(port);
    assert_eq!(field(&done, "state").as_str(), Some("done"), "{done}");
    let l = losses(&done);
    assert_eq!(l.len(), 60);
    assert!(l[59] < 0.5 * l[0], "loss {} -> {}", l[0], l[59]);
    let g = post(
        port,
        "/api/generate",
        r#"{"prompt":"the quick brown ","max_new":12}"#,
    );
    assert_eq!(g.status, 200, "{}", g.text());
    let out = g.json();
    let words = field(&out, "text").as_str().unwrap().to_string();
    assert!(words.starts_with("fox"), "got {words:?}");
    assert_eq!(
        field(&out, "tokens").to_string().matches(',').count(),
        11,
        "12 tokens"
    );
}
