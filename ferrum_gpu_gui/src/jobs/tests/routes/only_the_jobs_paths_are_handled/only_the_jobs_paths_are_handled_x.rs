use super::*;

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
