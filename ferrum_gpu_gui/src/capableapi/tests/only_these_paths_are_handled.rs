use super::*;

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
