use super::*;

#[test]
fn tools_are_listed_with_a_path_or_the_reason_they_are_missing() {
    let a = app();
    let (s, j) = call(&a, "GET", "/api/tools", b"");
    assert_eq!(s, 200);
    let Json::Arr(tools) = field(&j, "tools") else {
        panic!("{j}")
    };
    let names: Vec<_> = tools
        .iter()
        .map(|t| field(t, "name").as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["ollama", "lms"]);
    for t in &tools {
        // exactly one of path and note is set
        let (p, n) = (field(t, "path"), field(t, "note"));
        assert!(
            matches!(
                (&p, &n),
                (Json::Str(_), Json::Null) | (Json::Null, Json::Str(_))
            ),
            "{t}"
        );
        if let Json::Str(note) = n {
            assert!(
                note.contains("cannot find") || note.contains("no such file"),
                "{note}"
            );
        }
    }
}

#[test]
fn other_methods_and_unknown_routes_are_refused() {
    let a = app();
    assert_eq!(call(&a, "GET", "/api/serve/start", b"").0, 405);
    assert_eq!(call(&a, "POST", "/api/serve/status", b"").0, 405);
    assert_eq!(call(&a, "POST", "/api/tools", b"").0, 405);
    assert_eq!(call(&a, "GET", "/api/serve/nope", b"").0, 404);
}
