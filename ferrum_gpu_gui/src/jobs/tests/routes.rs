use super::support::{jobs, stub, wait};
use crate::app::{App, DeviceChoice};
use crate::engine::CpuBackend;
use crate::http::Request;
use crate::jobs::routes::{handles, route};
use crate::json::Json;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn app() -> Arc<App> {
    let (_dir, p) = stub();
    let a = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    a.jobs.set_cli(Some(p));
    a
}

fn call(a: &App, method: &str, path: &str, query: &[(&str, &str)], body: &[u8]) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: body.to_vec(),
    };
    let r = route(a, &req);
    (
        r.status,
        Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null),
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

fn wait_state(a: &App, id: u64, want: &str) -> Json {
    let t0 = Instant::now();
    loop {
        let (s, j) = call(a, "GET", &format!("/api/jobs/{id}"), &[], b"");
        assert_eq!(s, 200);
        if field(&j, "state").as_str() == Some(want) {
            return j;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "never {want}: {j}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn only_the_jobs_paths_are_handled() {
    assert!(handles("/api/jobs") && handles("/api/jobs/3") && handles("/api/jobs/3/stop"));
    assert!(!handles("/api/jobsx") && !handles("/api/job") && !handles("/api/train"));
}

#[test]
fn a_job_is_started_polled_and_listed() {
    let a = app();
    let (s, j) = call(
        &a,
        "POST",
        "/api/jobs",
        &[],
        br#"{"cmd":"info","args":["--device","cpu"]}"#,
    );
    assert_eq!(s, 200, "{j}");
    let id = field(&j, "id").as_usize().unwrap() as u64;
    let done = wait_state(&a, id, "done");
    assert_eq!(field(&done, "exit").as_f64(), Some(0.0));
    let Json::Arr(lines) = field(&done, "lines") else {
        panic!("{done}")
    };
    assert_eq!(field(&lines[0], "t").as_str(), Some("args: --device cpu"));
    assert_eq!(field(&lines[0], "e"), Json::Bool(false));
    assert_eq!(field(&done, "next").as_f64(), Some(2.0));
    // since skips what was seen
    let (_, tail) = call(
        &a,
        "GET",
        &format!("/api/jobs/{id}"),
        &[("since", "1")],
        b"",
    );
    let Json::Arr(l) = field(&tail, "lines") else {
        panic!()
    };
    assert_eq!(l.len(), 1);
    let (s, list) = call(&a, "GET", "/api/jobs", &[], b"");
    assert_eq!(s, 200);
    let Json::Arr(items) = field(&list, "jobs") else {
        panic!("{list}")
    };
    assert_eq!(items.len(), 1);
    assert_eq!(field(&items[0], "cmd").as_str(), Some("info"));
}

#[test]
fn progress_events_reach_the_page() {
    let a = app();
    let dir = std::env::temp_dir();
    let out = dir.join(format!("ferrum_route_{}.gguf", std::process::id()));
    let body = format!(r#"{{"cmd":"train","args":["--out","{}"]}}"#, out.display());
    let (_, j) = call(&a, "POST", "/api/jobs", &[], body.as_bytes());
    let id = field(&j, "id").as_usize().unwrap() as u64;
    let t0 = Instant::now();
    let seen = loop {
        let (_, p) = call(&a, "GET", &format!("/api/jobs/{id}"), &[], b"");
        if field(&field(&p, "step"), "step").as_f64() == Some(20.0) {
            break p;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "{p}");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(field(&field(&seen, "step"), "loss").as_f64(), Some(2.0));
    assert_eq!(field(&field(&seen, "val"), "best"), Json::Bool(true));
    let (s, r) = call(
        &a,
        "POST",
        &format!("/api/jobs/{id}/stop"),
        &[],
        br#"{"force":true}"#,
    );
    assert_eq!((s, field(&r, "stopping")), (200, Json::Bool(true)));
    wait_state(&a, id, "stopped");
    let (_, again) = call(&a, "POST", &format!("/api/jobs/{id}/stop"), &[], b"");
    assert_eq!(
        field(&again, "stopping"),
        Json::Bool(false),
        "already finished"
    );
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
    // no cli found: an error, not a crash
    let none = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    none.jobs.set_cli(Some("/no/such/ferrum-gpu".into()));
    let (s, j) = call(&none, "POST", "/api/jobs", &[], br#"{"cmd":"info"}"#);
    assert_eq!(s, 400);
    assert!(field(&j, "error")
        .as_str()
        .unwrap()
        .contains("no such file"));
    // an empty body has no cmd either
    assert_eq!(call(&a, "POST", "/api/jobs", &[], b"").0, 400);
    // null args are an empty list
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

#[test]
fn the_api_router_sends_jobs_paths_here() {
    let (j, _d) = jobs();
    let a = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    drop(j);
    let req = Request {
        method: "GET".into(),
        path: "/api/jobs".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &req).status, 200);
    let _ = wait; // the shared waiter is used by the other tests in this module
}

#[test]
fn the_cli_flag_is_parsed() {
    let o = crate::server::parse_args(&["--cli".into(), "/x/ferrum-gpu".into()]).unwrap();
    assert_eq!(
        o.cli.as_deref(),
        Some(std::path::Path::new("/x/ferrum-gpu"))
    );
    assert!(crate::server::parse_args(&["--cli".into()]).is_err());
    assert_eq!(crate::server::parse_args(&[]).unwrap().cli, None);
}
