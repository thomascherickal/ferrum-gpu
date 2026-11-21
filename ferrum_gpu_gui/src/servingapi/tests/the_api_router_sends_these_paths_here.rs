use super::*;

#[test]
fn the_api_router_sends_these_paths_here() {
    let a = app();
    for p in ["/api/serve/status", "/api/tools"] {
        let req = Request {
            method: "GET".into(),
            path: p.into(),
            query: vec![],
            headers: vec![],
            body: vec![],
        };
        assert_eq!(crate::api::route(&a, &req).status, 200, "{p}");
    }
}

#[test]
fn a_tool_is_reported_by_its_path_or_by_why_it_was_not_found() {
    let found = tool_json("lms", Ok("/usr/bin/lms".into()));
    assert_eq!(field(&found, "path").as_str(), Some("/usr/bin/lms"));
    assert_eq!(field(&found, "note"), Json::Null);
    let missing = tool_json("ollama", Err("not installed".into()));
    assert_eq!(field(&missing, "path"), Json::Null);
    assert_eq!(field(&missing, "note").as_str(), Some("not installed"));
    assert_eq!(field(&missing, "name").as_str(), Some("ollama"));
}
