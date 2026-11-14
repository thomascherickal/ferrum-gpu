//! Routes and refusals the journey tests do not reach.

use super::support::{app, app_with_model, call, call_raw, corpus, field, wait};
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;

#[test]
fn a_model_cannot_be_replaced_while_a_run_is_going() {
    let a = app_with_model();
    // a long run, so it is still going when the upload arrives
    let body = Json::obj([
        ("text", Json::Str(corpus())),
        ("steps", Json::Num(100_000.0)),
        ("batch", Json::Num(8.0)),
        ("context", Json::Num(24.0)),
    ])
    .to_string();
    assert_eq!(call(&a, "POST", "/api/train", &body).0, 200);
    let (s, _) = call_raw(&a, "PUT", "/api/model.gguf", &[], b"not even gguf");
    assert_eq!(s, 409, "refused before the bytes are looked at");
    let (s, j) = call(&a, "POST", "/api/model", r#"{"preset":"tiny"}"#);
    assert_eq!(s, 409, "{j}");
    // stop it, and the upload is judged on its bytes again
    assert_eq!(call(&a, "POST", "/api/train/stop", "").0, 200);
    let done = wait(&a);
    assert_eq!(field(&done, "state").as_str(), Some("stopped"), "{done}");
    assert_eq!(
        call_raw(&a, "PUT", "/api/model.gguf", &[], b"not even gguf").0,
        400
    );
}

#[test]
fn numeric_fields_out_of_range_are_400s_that_name_the_field() {
    let a = app_with_model();
    for (path, body, field_name) in [
        ("/api/model", r#"{"seed":-1}"#, "seed"),
        ("/api/model", r#"{"seed":"one"}"#, "seed"),
        ("/api/generate", r#"{"prompt":"hi","max_new":0}"#, "max_new"),
        (
            "/api/generate",
            r#"{"prompt":"hi","temperature":99}"#,
            "temperature",
        ),
        ("/api/generate", r#"{"prompt":"hi","top_k":-5}"#, "top_k"),
    ] {
        let (s, j) = call(&a, "POST", path, body);
        assert_eq!(s, 400, "{path} {body}: {j}");
        let msg = field(&j, "error");
        assert!(msg.as_str().unwrap().contains(field_name), "{body}: {msg}");
    }
}

#[test]
fn the_bench_model_route_serves_a_gguf_and_refuses_unknown_shapes() {
    let a = app();
    let (s, bytes) = call_raw(
        &a,
        "GET",
        "/api/selftest/bench-model.gguf",
        &[("shape", "small"), ("quant", "q8_0")],
        b"",
    );
    assert_eq!(s, 200);
    assert_eq!(&bytes[..4], b"GGUF");
    let (s, bytes) = call_raw(
        &a,
        "GET",
        "/api/selftest/bench-model.gguf",
        &[("shape", "huge")],
        b"",
    );
    assert_eq!(s, 400);
    assert!(String::from_utf8_lossy(&bytes).contains("unknown shape 'huge'"));
    let (s, bytes) = call_raw(
        &a,
        "GET",
        "/api/selftest/bench-model.gguf",
        &[("quant", "q2")],
        b"",
    );
    assert_eq!(s, 400);
    assert!(String::from_utf8_lossy(&bytes).contains("unknown quant 'q2'"));
}

#[test]
fn unknown_routes_and_methods_are_404_and_405_never_a_crash() {
    let a = app();
    assert_eq!(call(&a, "GET", "/api/nope", "").0, 404);
    assert_eq!(call(&a, "GET", "/nope", "").0, 404);
    assert_eq!(call(&a, "DELETE", "/api/info", "").0, 405);
    assert_eq!(call(&a, "POST", "/index.html", "").0, 405);
}

#[test]
fn a_started_server_reports_its_url() {
    let a = app();
    let server = start(a, 0).unwrap();
    let url = server.url();
    assert_eq!(url, format!("http://127.0.0.1:{}/", server.addr.port()));
    server.stop();
}

#[test]
fn a_tied_model_is_created_and_reported_but_training_it_is_refused_by_name() {
    let a = app();
    let (s, j) = call(&a, "POST", "/api/model", r#"{"preset":"tiny","tied":true}"#);
    assert_eq!(s, 200, "{j}");
    assert_eq!(field(&j, "tied"), Json::Bool(true));
    // the GUI's trainer needs an untied head (the engine says so); a tied model still decodes and exports
    let (s, g) = call(
        &a,
        "POST",
        "/api/generate",
        r#"{"prompt":"hi","max_new":3}"#,
    );
    assert_eq!(s, 200, "{g}");
    let body = Json::obj([
        ("text", Json::Str(corpus())),
        ("steps", Json::Num(5.0)),
        ("batch", Json::Num(4.0)),
        ("context", Json::Num(16.0)),
    ])
    .to_string();
    assert_eq!(call(&a, "POST", "/api/train", &body).0, 200);
    let done = wait(&a);
    assert_eq!(field(&done, "state").as_str(), Some("error"), "{done}");
    assert!(
        field(&done, "error")
            .as_str()
            .unwrap()
            .contains("tied head"),
        "{done}"
    );
    let (s, g) = call(&a, "GET", "/api/info", "");
    assert_eq!(s, 200);
    assert_eq!(field(&field(&g, "model"), "tied"), Json::Bool(true));
    let (_, untied) = call(&a, "POST", "/api/model", r#"{"preset":"tiny"}"#);
    assert_eq!(field(&untied, "tied"), Json::Bool(false), "off by default");
}

#[test]
fn tied_must_be_a_boolean() {
    let a = app();
    for body in [r#"{"tied":"yes"}"#, r#"{"tied":1}"#] {
        let (s, j) = call(&a, "POST", "/api/model", body);
        assert_eq!(s, 400, "{body}: {j}");
        assert!(field(&j, "error").as_str().unwrap().contains("tied"));
    }
}

#[test]
fn the_page_offers_the_tied_option_and_the_script_sends_it() {
    let a = app();
    let page = call_raw(&a, "GET", "/", &[], b"").1;
    assert!(String::from_utf8_lossy(&page).contains(r#"id="tied""#));
    let js = call_raw(&a, "GET", "/app.js", &[], b"").1;
    assert!(String::from_utf8_lossy(&js).contains("tied: $('tied').checked"));
}
