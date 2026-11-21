use super::*;

#[test]
fn no_manifest_points_at_ferrum_cpu_or_outside() {
    let root = root();
    let mut bad = vec![];
    for m in files("", Some("Cargo.toml")) {
        let text = fs::read_to_string(&m).unwrap();
        for v in manifest_violations(&text, m.parent().unwrap(), &root) {
            bad.push(format!("{}: {v}", m.display()));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn no_source_reaches_ferrum_cpu() {
    let bad = scan("rs", None, &|t, d| source_violations(t, d, &root()));
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn no_unsafe_anywhere() {
    let bad = scan("rs", None, &|t, _| unsafe_lines(t));
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn the_detectors_catch_what_they_should() {
    let (root, dir) = (Path::new("/r"), Path::new("/r/crate"));
    let mv = |t: &str| manifest_violations(t, dir, root);
    assert!(!mv("ferrum_core = { path = \"../ferrum-cpu/ferrum_core\" }").is_empty());
    assert!(!mv("x = { path = \"../../other\" }").is_empty());
    assert!(!mv("x = { git = \"https://e/x\" }").is_empty());
    assert!(mv("x = { path = \"../sibling\" }").is_empty());
    assert!(mv("x = { path = \"./inner\" }").is_empty());
    assert!(mv("# ferrum-cpu mention\nwgpu = \"30\"").is_empty());
    assert_eq!(
        path_value("x = { path = \"../a\", version = \"1\" }").as_deref(),
        Some("../a")
    );
    assert_eq!(path_value("wgpu = \"30\""), None);
    let sv = |t: &str| source_violations(t, dir, root);
    assert!(!sv("use ferrum_core::Tensor;").is_empty());
    assert!(!sv("let s = include_str!(\"../../ferrum-cpu/x\");").is_empty());
    assert!(
        !sv("let s = include_str!(\"../../other/x\");").is_empty(),
        "leaves the repo"
    );
    assert!(sv("// from ferrum_core::ops (provenance)").is_empty());
    assert!(sv("let s = include_str!(\"shaders/a.wgsl\");").is_empty());
    assert!(
        sv("let s = include_str!(\"../shaders/a.wgsl\");").is_empty(),
        "inside the repo"
    );
    assert!(!unsafe_lines("unsafe { x() }").is_empty());
    assert!(unsafe_lines("#![forbid(unsafe_code)]").is_empty());
}
