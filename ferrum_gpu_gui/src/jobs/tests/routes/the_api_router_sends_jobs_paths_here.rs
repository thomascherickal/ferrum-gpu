use super::*;

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
