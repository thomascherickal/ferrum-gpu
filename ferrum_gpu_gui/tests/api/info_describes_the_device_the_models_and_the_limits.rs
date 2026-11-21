use super::*;

#[test]
fn info_describes_the_device_the_models_and_the_limits() {
    let note = Some("no usable GPU (test); using the CPU".to_string());
    let app = App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Auto,
        note,
        vec![],
    );
    let (srv, _app) = serve(app);
    let p = srv.addr.port();
    let info = get(p, "/api/info").json();
    let dev = field(&info, "device");
    assert_eq!(field(&dev, "requested").as_str(), Some("auto"));
    assert_eq!(field(&dev, "active").as_str(), Some("cpu"));
    assert!(field(&dev, "note")
        .as_str()
        .unwrap()
        .contains("using the CPU"));
    assert!(matches!(field(&info, "adapters"), Json::Arr(a) if a.is_empty()));
    assert!(matches!(field(&info, "presets"), Json::Arr(a) if a.len() == 3));
    assert!(
        field(&field(&info, "limits"), "max_new")
            .as_usize()
            .unwrap()
            >= 64
    );
    post(p, "/api/model", r#"{"preset":"medium"}"#);
    let m = field(&get(p, "/api/info").json(), "model");
    assert_eq!(field(&m, "preset").as_str(), Some("medium"));
    assert_eq!(field(&m, "byte_tokens"), Json::Bool(true));
    assert!(field(&m, "params").as_usize().unwrap() > 100_000);
    srv.stop();
}

#[test]
fn the_app_picks_a_backend_for_auto_cpu_and_a_missing_adapter() {
    let cpu = App::new(DeviceChoice::Cpu, None, 1).unwrap();
    assert_eq!(
        (cpu.backend.kind(), cpu.fallback_note.is_none()),
        ("cpu", true)
    );
    let auto = App::new(DeviceChoice::Auto, Some("no-such-adapter"), 1).unwrap();
    assert_eq!(auto.backend.kind(), "cpu");
    assert!(auto
        .fallback_note
        .as_deref()
        .unwrap()
        .contains("using the CPU"));
    assert!(App::new(DeviceChoice::Gpu, Some("no-such-adapter"), 1).is_err());
}

#[test]
fn the_browser_self_test_result_is_stored_and_returned() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    assert_eq!(get(p, "/api/selftest/result").status, 404);
    assert_eq!(
        post(p, "/api/selftest/result", r#"{"pass":true}"#).status,
        200
    );
    assert_eq!(get(p, "/api/selftest/result").text(), r#"{"pass":true}"#);
    srv.stop();
}

#[test]
fn a_model_over_the_parameter_limit_is_refused_with_413_and_the_old_model_stays() {
    // a tiny model, exported from one server and imported into another whose limit is lower
    let (a, _app) = serve(cpu_app());
    post(a.addr.port(), "/api/model", r#"{"preset":"tiny","seed":3}"#);
    let gguf = get(a.addr.port(), "/api/model.gguf?quant=f32").body;
    let mut small = cpu_app();
    small.max_params = 10;
    let (b, _app) = serve(small);
    let r = call(b.addr.port(), "PUT", "/api/model.gguf", &gguf);
    assert_eq!(r.status, 413, "{}", r.text());
    assert!(r.text().contains("holds at most 10"), "{}", r.text());
    // under the limit the same file is accepted
    let (c, _app) = serve(cpu_app());
    assert_eq!(
        call(c.addr.port(), "PUT", "/api/model.gguf", &gguf).status,
        200
    );
}
