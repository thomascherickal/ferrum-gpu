use super::*;

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
