use super::*;

#[test]
fn a_failing_command_is_an_error_job_with_the_message_as_a_line() {
    let a = app();
    let (_, j) = call(&a, "POST", "/api/jobs", &[], br#"{"cmd":"eval"}"#);
    let id = field(&j, "id").as_usize().unwrap() as u64;
    let p = wait_state(&a, id, "error");
    assert_eq!(field(&p, "exit").as_f64(), Some(2.0));
    let Json::Arr(lines) = field(&p, "lines") else {
        panic!("{p}")
    };
    assert_eq!(
        field(&lines[0], "t").as_str(),
        Some("the fake command failed")
    );
    assert_eq!(field(&lines[0], "e"), Json::Bool(true));
}

#[test]
fn bad_start_requests_are_400s_that_say_why() {
    let a = app();
    for (body, needle) in [
        (&br#"{"args":[]}"#[..], "cmd is required"),
        (br#"{"cmd":"sh"}"#, "not a command"),
        (br#"{"cmd":"info","args":"x"}"#, "list of strings"),
        (br#"{"cmd":"info","args":[1]}"#, "list of strings"),
        (
            br#"{"cmd":"export-ollama","args":["--bin","/bin/sh"]}"#,
            "ollama or lms",
        ),
        (b"{not json", "bad JSON"),
        (&[0xff, 0xfe][..], "UTF-8"),
    ] {
        let (s, j) = call(&a, "POST", "/api/jobs", &[], body);
        assert_eq!(s, 400, "{}: {j}", String::from_utf8_lossy(body));
        let msg = field(&j, "error");
        assert!(msg.as_str().unwrap().contains(needle), "{msg}");
    }
    assert_eq!(call(&a, "POST", "/api/jobs", &[], b"").0, 400);
    let (s, _) = call(
        &a,
        "POST",
        "/api/jobs",
        &[],
        br#"{"cmd":"info","args":null}"#,
    );
    assert_eq!(s, 200);
}

#[test]
fn unknown_jobs_routes_and_methods_are_refused() {
    let a = app();
    assert_eq!(call(&a, "GET", "/api/jobs/99", &[], b"").0, 404);
    assert_eq!(call(&a, "GET", "/api/jobs/abc", &[], b"").0, 404);
    assert_eq!(call(&a, "DELETE", "/api/jobs", &[], b"").0, 405);
    let (s, j) = call(&a, "POST", "/api/jobs", &[], br#"{"cmd":"info"}"#);
    assert_eq!(s, 200);
    let id = field(&j, "id").as_usize().unwrap() as u64;
    let base = format!("/api/jobs/{id}");
    assert_eq!(call(&a, "PUT", &base, &[], b"").0, 405);
    assert_eq!(call(&a, "GET", &format!("{base}/stop"), &[], b"").0, 405);
    assert_eq!(call(&a, "GET", &format!("{base}/zzz"), &[], b"").0, 404);
    assert_eq!(call(&a, "GET", &base, &[("since", "x")], b"").0, 400);
    assert_eq!(
        call(&a, "POST", &format!("{base}/stop"), &[], b"{bad").0,
        400
    );
}
