use super::*;

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
    assert!(field(&j, "ram_total").as_f64().unwrap() > 0.0);
    assert!(field(&ads[0], "budget_mib").as_f64().unwrap() > 0.0);
    let fits = field(&ads[0], "fits");
    let Json::Arr(infer) = field(&fits, "infer") else {
        panic!("{ads:?}")
    };
    assert_eq!(infer.len(), 3);
    assert!(field(&fits, "train").as_str().is_some());
    assert!(
        field(&ads[0], "note").as_str().unwrap().contains("budget")
            || field(&ads[0], "note").as_str().unwrap().contains("unknown")
    );
    // no adapters is an empty list, not an error
    let (_, none) = get(&app(vec![]), "/api/system", &[]);
    assert_eq!(field(&none, "adapters"), Json::Arr(vec![]));
}
