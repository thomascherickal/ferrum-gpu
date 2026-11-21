use super::*;

#[test]
fn only_the_serve_and_tools_paths_are_handled() {
    assert!(handles("/api/serve/status") && handles("/api/serve/start") && handles("/api/tools"));
    assert!(!handles("/api/serve") && !handles("/api/toolsx") && !handles("/api/jobs"));
}

#[test]
fn an_idle_slot_reports_not_running_and_stop_has_nothing_to_do() {
    let a = app();
    let (s, j) = call(&a, "GET", "/api/serve/status", b"");
    assert_eq!(s, 200);
    assert_eq!(field(&j, "running"), Json::Bool(false));
    assert_eq!(field(&j, "addr"), Json::Null);
    assert_eq!(
        call(&a, "POST", "/api/serve/stop", b"").1.get("stopped"),
        Some(&Json::Bool(false))
    );
}

#[test]
fn bad_start_requests_are_400s_that_say_why() {
    let a = app();
    for (body, needle) in [
        (&br#"{"args":"x"}"#[..], "list of strings"),
        (br#"{"args":[1]}"#, "list of strings"),
        (br#"{}"#, "list of strings"),
        (b"{nope", "bad JSON"),
        (&[0xff, 0xfe][..], "UTF-8"),
        (br#"{"args":["--model","a\u0000b"]}"#, "NUL"),
        (br#"{"args":["--port","0"]}"#, "--model"),
        (
            br#"{"args":["--model","/no/such/model.gguf","--port","0","--device","cpu"]}"#,
            "no/such/model.gguf",
        ),
        (br#"{"args":["--nope"]}"#, "--nope"),
    ] {
        let (s, j) = call(&a, "POST", "/api/serve/start", body);
        assert_eq!(s, 400, "{}: {j}", String::from_utf8_lossy(body));
        let msg = field(&j, "error");
        assert!(msg.as_str().unwrap().contains(needle), "{msg}");
    }
    // the failed attempts are in the status lines, and nothing is running
    let (_, st) = call(&a, "GET", "/api/serve/status", b"");
    assert_eq!(field(&st, "running"), Json::Bool(false));
}
