use super::*;

#[test]
fn each_adapter_lists_its_features_by_name() {
    let f16 = AdapterFeatures {
        shader_f16: true,
        subgroups: true,
        ..Default::default()
    };
    let a = app_with(vec![
        adapter("Fake Discrete GPU", Kind::Discrete, f16),
        adapter("Fake Plain GPU", Kind::Discrete, AdapterFeatures::default()),
    ]);
    let (s, j) = call(&a, "GET", "/api/info", "");
    assert_eq!(s, 200);
    let ads = adapters_of(&j);
    let feats = |i: usize| match field(&ads[i], "features") {
        Json::Arr(v) => v
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        other => panic!("{other}"),
    };
    assert_eq!(feats(0), ["f16", "subgroups"]);
    assert!(feats(1).is_empty());
}

#[test]
fn a_discrete_adapter_no_tool_knows_has_an_unknown_budget_that_says_how_to_set_one() {
    // the name matches no nvidia-smi entry, so there is no free-memory figure to go on
    let a = app_with(vec![adapter(
        "Fake Discrete GPU",
        Kind::Discrete,
        AdapterFeatures::default(),
    )]);
    let (_, j) = call(&a, "GET", "/api/info", "");
    let ad = &adapters_of(&j)[0];
    assert_eq!(field(ad, "memory_budget_mib"), Json::Null);
    let note = field(ad, "memory_note");
    let note = note.as_str().unwrap();
    assert!(
        note.contains("unknown") && note.contains("--gpu-mem-mb"),
        "{note}"
    );
}

#[test]
fn an_integrated_adapter_gets_a_ram_based_budget_on_linux() {
    if !cfg!(target_os = "linux") {
        eprintln!("SKIP: the RAM figure is read from /proc/meminfo");
        return;
    }
    let a = app_with(vec![adapter(
        "Fake Integrated GPU",
        Kind::Integrated,
        AdapterFeatures::default(),
    )]);
    let (_, j) = call(&a, "GET", "/api/info", "");
    let ad = &adapters_of(&j)[0];
    let mib = field(ad, "memory_budget_mib").as_f64().expect("a number");
    assert!(mib > 0.0);
    let note = field(ad, "memory_note");
    let note = note.as_str().unwrap();
    assert!(note.contains("70%") && note.contains("estimate"), "{note}");
}

#[test]
fn the_device_table_has_columns_for_both() {
    let a = app_with(vec![]);
    let html =
        String::from_utf8_lossy(&crate::support::call_raw(&a, "GET", "/", &[], b"").1).into_owned();
    assert!(html.contains("<th scope=\"col\">Features</th>"));
    assert!(html.contains("Memory budget (estimate)"));
    let js = String::from_utf8_lossy(&crate::support::call_raw(&a, "GET", "/app.js", &[], b"").1)
        .into_owned();
    assert!(js.contains("memory_budget_mib") && js.contains("a.features"));
}
