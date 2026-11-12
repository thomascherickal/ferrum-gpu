//! Helpers: an app on the CPU backend, in-process API calls, and a training run to completion.

use ferrum_gpu_gui::api::route;
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::http::Request;
use ferrum_gpu_gui::json::Json;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn app() -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ))
}

/// Call the API; the status and the JSON body.
pub fn call(app: &Arc<App>, method: &str, path: &str, body: &str) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: vec![],
        headers: vec![],
        body: body.as_bytes().to_vec(),
    };
    let r = route(app, &req);
    let text = String::from_utf8_lossy(&r.body).into_owned();
    (r.status, Json::parse(&text).unwrap_or(Json::Null))
}

pub fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

/// The training text used throughout: a short phrase, repeated.
pub fn corpus() -> String {
    "the quick brown fox jumps over the lazy dog. ".repeat(14)
}

/// A fresh tiny model on a fresh app.
pub fn app_with_model() -> Arc<App> {
    let a = app();
    let (s, _) = call(&a, "POST", "/api/model", r#"{"preset":"tiny","seed":7}"#);
    assert_eq!(s, 200);
    a
}

/// `/api/train` with `extra` merged over the default request, waited to the end; the first reply
/// and the final status.
pub fn train(app: &Arc<App>, val: Option<f64>) -> ((u16, Json), Json) {
    let mut members = vec![
        ("text", Json::Str(corpus())),
        ("steps", Json::Num(60.0)),
        ("lr", Json::Num(0.01)),
        ("batch", Json::Num(4.0)),
        ("context", Json::Num(24.0)),
    ];
    if let Some(v) = val {
        members.push(("val", Json::Num(v)));
    }
    let body = Json::Obj(members.into_iter().map(|(k, v)| (k.into(), v)).collect());
    let reply = call(app, "POST", "/api/train", &body.to_string());
    (reply.clone(), wait(app))
}

/// Wait until no run is going; the status.
pub fn wait(app: &Arc<App>) -> Json {
    let t0 = Instant::now();
    loop {
        let (_, s) = call(app, "GET", "/api/train/status", "");
        if field(&s, "state").as_str() != Some("running") {
            return s;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "training did not finish"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A trained app: 60 steps, no validation share.
pub fn trained() -> Arc<App> {
    let a = app_with_model();
    let (first, done) = train(&a, None);
    assert_eq!(first.0, 200);
    assert_eq!(field(&done, "state").as_str(), Some("done"), "{done}");
    a
}
