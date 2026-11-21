use super::*;

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

#[test]
fn an_unknown_capable_path_and_a_post_are_refused() {
    let a = app(vec![]);
    assert_eq!(get(&a, "/api/capable/other", &[]).0, 404);
    let post = Request {
        method: "POST".into(),
        path: "/api/capable".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(route(&a, &post).status, 405);
}
