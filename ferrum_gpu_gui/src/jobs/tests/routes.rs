use crate::app::{App, DeviceChoice};
use crate::engine::CpuBackend;
use crate::http::Request;
use crate::jobs::routes::{handles, route};
use crate::jobs::{Jobs, RunFn};
use crate::json::Json;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Fake commands: `info` echoes its arguments, `train` reports progress then waits to be stopped,
/// `eval` fails.
fn fake() -> Jobs {
    let run: Arc<RunFn> = Arc::new(|argv, out| {
        match argv[0].as_str() {
            "info" => {
                writeln!(out, "args: {}", argv[1..].join(" ")).unwrap();
                writeln!(out, "device: fake").unwrap();
            }
            "train" => {
                writeln!(
                    out,
                    "step 10/100 loss 3.0000 grad_norm 0.500 tokens/s 1000 elapsed 1.0s eta 9s"
                )
                .unwrap();
                writeln!(out, "epoch 1 val loss 2.5000 (best)").unwrap();
                writeln!(
                    out,
                    "step 20/100 loss 2.0000 grad_norm 0.400 tokens/s 1100 elapsed 2.0s eta 8s"
                )
                .unwrap();
                let t0 = Instant::now();
                while !ferrum_gpu_core::cancel::requested() {
                    assert!(t0.elapsed() < Duration::from_secs(20), "never stopped");
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            _ => return Err("the fake command failed".into()),
        }
        Ok(())
    });
    Jobs::with_runner(run)
}

fn app() -> Arc<App> {
    Arc::new(
        App::with_backend(
            Arc::new(CpuBackend::new(1)),
            DeviceChoice::Cpu,
            None,
            vec![],
        )
        .with_jobs(fake()),
    )
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
        std::thread::sleep(Duration::from_millis(5));
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
fn progress_events_reach_the_page_and_stop_ends_the_job() {
    let a = app();
    let (_, j) = call(
        &a,
        "POST",
        "/api/jobs",
        &[],
        br#"{"cmd":"train","args":["--out","m.gguf"]}"#,
    );
    let id = field(&j, "id").as_usize().unwrap() as u64;
    let t0 = Instant::now();
    let seen = loop {
        let (_, p) = call(&a, "GET", &format!("/api/jobs/{id}"), &[], b"");
        if field(&field(&p, "step"), "step").as_f64() == Some(20.0) {
            break p;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "{p}");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(field(&field(&seen, "step"), "loss").as_f64(), Some(2.0));
    assert_eq!(field(&field(&seen, "val"), "best"), Json::Bool(true));
    for body in [&br#"{"force":true}"#[..], b""] {
        let (s, r) = call(&a, "POST", &format!("/api/jobs/{id}/stop"), &[], body);
        assert_eq!((s, field(&r, "stopping")), (200, Json::Bool(true)));
    }
    wait_state(&a, id, "stopped");
    let (_, again) = call(&a, "POST", &format!("/api/jobs/{id}/stop"), &[], b"");
    assert_eq!(
        field(&again, "stopping"),
        Json::Bool(false),
        "already finished"
    );
}

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

#[test]
fn the_api_router_sends_jobs_paths_here() {
    let a = app();
    let req = Request {
        method: "GET".into(),
        path: "/api/jobs".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &req).status, 200);
}

#[test]
fn the_default_app_runs_the_real_commands() {
    let a = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let (_, j) = call(
        &a,
        "POST",
        "/api/jobs",
        &[],
        br#"{"cmd":"info","args":["--device","cpu"]}"#,
    );
    let id = field(&j, "id").as_usize().unwrap() as u64;
    let done = wait_state(&a, id, "done");
    assert_eq!(field(&done, "exit").as_f64(), Some(0.0), "{done}");
}
