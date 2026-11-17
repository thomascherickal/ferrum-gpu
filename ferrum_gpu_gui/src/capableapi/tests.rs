use super::*;
use crate::app::DeviceChoice;
use crate::engine::CpuBackend;
use ferrum_wgpu::select::{AdapterFeatures, AdapterSummary, Kind};
use std::sync::Arc;

fn app(adapters: Vec<AdapterSummary>) -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        adapters,
    ))
}

fn get(a: &App, path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    let req = Request {
        method: "GET".into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: vec![],
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

#[test]
fn only_these_paths_are_handled() {
    assert!(handles("/api/capable") && handles("/api/system"));
    assert!(!handles("/api/capables") && !handles("/api/serve/status"));
}

#[test]
fn a_memory_size_gets_the_documented_answer() {
    let a = app(vec![]);
    let (s, j) = get(&a, "/api/capable", &[("memory_gb", "16")]);
    assert_eq!(s, 200, "{j}");
    let row = field(&j, "row");
    let Json::Arr(infer) = field(&row, "infer") else {
        panic!("{j}")
    };
    assert_eq!(field(&infer[0], "text").as_str(), Some("22.8B"));
    assert_eq!(field(&field(&row, "train"), "text").as_str(), Some("816M"));
    assert_eq!(field(&j, "ceiling"), Json::Null, "no speed asked for");
    // unified memory holds less
    let (_, u) = get(&a, "/api/capable", &[("memory_gb", "16"), ("unified", "1")]);
    let Json::Arr(ui) = field(&field(&u, "row"), "infer") else {
        panic!()
    };
    assert_eq!(field(&ui[0], "text").as_str(), Some("17.6B"));
}

#[test]
fn a_model_and_a_bandwidth_give_the_decode_ceilings() {
    let a = app(vec![]);
    let (_, j) = get(
        &a,
        "/api/capable",
        &[("memory_gb", "24"), ("bandwidth", "936"), ("model_b", "7")],
    );
    let Json::Arr(c) = field(&j, "ceiling") else {
        panic!("{j}")
    };
    assert_eq!(c.len(), 3);
    assert_eq!(field(&c[0], "precision").as_str(), Some("Q4_0"));
    let q4 = field(&c[0], "tokens_per_s").as_f64().unwrap();
    assert!((q4 - 237.7).abs() < 0.1, "{q4}");
    assert!(field(&c[2], "tokens_per_s").as_f64().unwrap() < q4 / 6.0);
    // one of the two alone gives no ceiling
    let (_, only_bw) = get(
        &a,
        "/api/capable",
        &[("memory_gb", "24"), ("bandwidth", "936")],
    );
    assert_eq!(field(&only_bw, "ceiling"), Json::Null);
}

#[test]
fn bad_inputs_are_400s_that_name_the_field() {
    let a = app(vec![]);
    for (q, needle) in [
        (vec![], "memory_gb"),
        (vec![("memory_gb", "")], "memory_gb"),
        (vec![("memory_gb", "0")], "memory_gb"),
        (vec![("memory_gb", "-4")], "memory_gb"),
        (vec![("memory_gb", "9999999")], "memory_gb"),
        (vec![("memory_gb", "abc")], "memory_gb must be a number"),
        (vec![("memory_gb", "inf")], "memory_gb"),
        (
            vec![("memory_gb", "8"), ("bandwidth", "fast"), ("model_b", "7")],
            "bandwidth must be a number",
        ),
        (
            vec![("memory_gb", "8"), ("bandwidth", "100"), ("model_b", "x")],
            "model_b must be a number",
        ),
        (
            vec![("memory_gb", "8"), ("bandwidth", "0"), ("model_b", "7")],
            "above 0",
        ),
        (
            vec![("memory_gb", "8"), ("bandwidth", "100"), ("model_b", "-1")],
            "above 0",
        ),
    ] {
        let (s, j) = get(&a, "/api/capable", &q);
        assert_eq!(s, 400, "{q:?}: {j}");
        assert!(
            field(&j, "error").as_str().unwrap().contains(needle),
            "{q:?}: {j}"
        );
    }
}

#[test]
fn the_system_reports_the_machine_and_what_each_adapters_budget_holds() {
    let a = app(vec![AdapterSummary {
        name: "Fake Integrated GPU".into(),
        backend: "Vulkan".into(),
        kind: Kind::Integrated,
        max_buffer: 1 << 32,
        max_binding: 2047 << 20,
        max_workgroup_invocations: 256,
        features: AdapterFeatures::default(),
    }]);
    let (s, j) = get(&a, "/api/system", &[]);
    assert_eq!(s, 200, "{j}");
    assert_eq!(field(&j, "os").as_str(), Some(std::env::consts::OS));
    assert!(field(&j, "cpu_threads").as_f64().unwrap() >= 1.0);
    let Json::Arr(ads) = field(&j, "adapters") else {
        panic!("{j}")
    };
    assert_eq!(field(&ads[0], "name").as_str(), Some("Fake Integrated GPU"));
    if cfg!(target_os = "linux") {
        assert!(field(&j, "ram_total").as_f64().unwrap() > 0.0);
        assert!(field(&ads[0], "budget_mib").as_f64().unwrap() > 0.0);
        let fits = field(&ads[0], "fits");
        let Json::Arr(infer) = field(&fits, "infer") else {
            panic!("{ads:?}")
        };
        assert_eq!(infer.len(), 3);
        assert!(field(&fits, "train").as_str().is_some());
    }
    assert!(
        field(&ads[0], "note").as_str().unwrap().contains("budget")
            || field(&ads[0], "note").as_str().unwrap().contains("unknown")
    );
    // no adapters is an empty list, not an error
    let (_, none) = get(&app(vec![]), "/api/system", &[]);
    assert_eq!(field(&none, "adapters"), Json::Arr(vec![]));
}

#[test]
fn other_methods_are_refused_and_the_router_sends_these_paths_here() {
    let a = app(vec![]);
    let post = Request {
        method: "POST".into(),
        path: "/api/capable".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(route(&a, &post).status, 405);
    let ok = Request {
        method: "GET".into(),
        path: "/api/system".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &ok).status, 200);
}
