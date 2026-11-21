use super::*;

#[test]
fn a_run_can_be_stopped_and_keeps_its_partial_weights() {
    let (srv, app) = serve(cpu_app());
    let p = srv.addr.port();
    post(p, "/api/model", r#"{"preset":"tiny"}"#);
    let before = ferrum_gpu_gui::app::lock(&app.model).clone().unwrap();
    let body = format!(
        r#"{{"text":"{}","steps":100000,"batch":8,"context":32}}"#,
        "ab".repeat(200)
    );
    assert_eq!(post(p, "/api/train", &body).status, 200);
    let t0 = Instant::now();
    while field(&get(p, "/api/train/status").json(), "step")
        .as_usize()
        .unwrap_or(0)
        < 3
    {
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        post(p, "/api/train/stop", "{}").json().get("stopping"),
        Some(&Json::Bool(true))
    );
    let s = wait_until_done(p);
    assert_eq!(field(&s, "state").as_str(), Some("stopped"));
    assert!(field(&s, "step").as_usize().unwrap() < 100_000);
    assert_ne!(
        ferrum_gpu_gui::app::lock(&app.model).clone().unwrap(),
        before,
        "the partial weights were kept"
    );
    srv.stop();
}

#[test]
fn a_failed_run_is_reported_and_the_model_is_untouched() {
    // a model whose head is tied cannot be trained: the run fails, with the reason
    let (srv, app) = serve(cpu_app());
    let p = srv.addr.port();
    let mut tied = ferrum_gpu_gui::engine::fresh_model("tiny", 1).unwrap();
    tied = ferrum_ref::llama::Llama { tied: true, ..tied };
    *ferrum_gpu_gui::app::lock(&app.model) = Some(tied.clone());
    let body = format!(r#"{{"text":"{}","steps":3}}"#, "ab".repeat(100));
    assert_eq!(post(p, "/api/train", &body).status, 200);
    let s = wait_until_done(p);
    assert_eq!(field(&s, "state").as_str(), Some("error"));
    assert!(field(&s, "error").as_str().unwrap().contains("tied"), "{s}");
    assert_eq!(ferrum_gpu_gui::app::lock(&app.model).clone().unwrap(), tied);
    srv.stop();
}
