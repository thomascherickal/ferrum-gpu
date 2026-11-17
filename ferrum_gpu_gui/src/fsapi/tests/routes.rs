use super::support::{put, tmp};
use crate::fsapi::routes::{handles, route};
use crate::http::Request;
use crate::json::Json;

fn get(path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    call("GET", path, query)
}

fn call(method: &str, path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: vec![],
    };
    let r = route(&req);
    (
        r.status,
        Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null),
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

#[test]
fn only_the_fs_paths_are_handled() {
    assert!(handles("/api/fs/list") && handles("/api/fs/stat") && handles("/api/fs/x"));
    assert!(!handles("/api/fs") && !handles("/api/jobs") && !handles("/fs/list"));
}

#[test]
fn a_listing_has_entries_a_parent_and_places_to_start() {
    let d = tmp();
    put(&d, "a.txt", b"abc");
    std::fs::create_dir(d.join("sub")).unwrap();
    let (s, j) = get("/api/fs/list", &[("dir", d.to_str().unwrap())]);
    assert_eq!(s, 200, "{j}");
    assert_eq!(field(&j, "dir").as_str(), d.to_str());
    assert_eq!(field(&j, "parent").as_str(), d.parent().unwrap().to_str());
    let Json::Arr(entries) = field(&j, "entries") else {
        panic!("{j}")
    };
    assert_eq!(field(&entries[0], "name").as_str(), Some("sub"));
    assert_eq!(field(&entries[0], "kind").as_str(), Some("dir"));
    assert_eq!(field(&entries[1], "size").as_f64(), Some(3.0));
    assert_eq!(field(&entries[1], "hidden"), Json::Bool(false));
    let Json::Arr(places) = field(&j, "places") else {
        panic!("{j}")
    };
    assert!(!places.is_empty());
    assert!(places
        .iter()
        .any(|p| field(p, "name").as_str() == Some("Temp")));
}

#[test]
fn the_default_listing_is_where_the_server_started() {
    let (s, j) = get("/api/fs/list", &[]);
    assert_eq!(s, 200);
    assert_eq!(
        field(&j, "dir").as_str().map(String::from),
        std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
    );
    assert_eq!(get("/api/fs/list", &[("dir", "")]).0, 200);
}

#[test]
fn listing_errors_are_400s() {
    let (s, j) = get("/api/fs/list", &[("dir", "/definitely/not/here")]);
    assert_eq!(s, 400);
    assert!(field(&j, "error").as_str().unwrap().contains("not/here"));
    assert_eq!(get("/api/fs/list", &[("dir", "a\0b")]).0, 400);
}

#[test]
fn stat_reports_files_folders_and_missing_paths() {
    let d = tmp();
    let f = put(&d, "x/y.txt", b"12345");
    let (_, j) = get("/api/fs/stat", &[("path", f.to_str().unwrap())]);
    assert_eq!(field(&j, "kind").as_str(), Some("file"));
    assert_eq!(field(&j, "bytes").as_f64(), Some(5.0));
    let (_, j) = get("/api/fs/stat", &[("path", d.to_str().unwrap())]);
    assert_eq!(
        (field(&j, "kind").as_str(), field(&j, "files").as_f64()),
        (Some("dir"), Some(1.0))
    );
    assert_eq!(field(&j, "truncated"), Json::Bool(false));
    let (s, j) = get("/api/fs/stat", &[("path", "/no/such/path")]);
    assert_eq!((s, field(&j, "kind").as_str()), (200, Some("missing")));
}

#[test]
fn stat_and_preview_need_a_path() {
    for route in ["/api/fs/stat", "/api/fs/preview"] {
        for q in [vec![], vec![("path", "")], vec![("path", "a\0b")]] {
            let (s, j) = get(route, &q);
            assert_eq!(s, 400, "{route} {q:?}");
            assert!(field(&j, "error")
                .as_str()
                .unwrap()
                .contains("path is required"));
        }
    }
}

#[test]
fn preview_returns_text_and_validates_bytes() {
    let d = tmp();
    let f = put(&d, "t.txt", b"hello\nworld\n");
    let p = f.to_str().unwrap();
    let (s, j) = get("/api/fs/preview", &[("path", p)]);
    assert_eq!(s, 200);
    assert_eq!(field(&j, "text").as_str(), Some("hello\nworld\n"));
    assert_eq!(field(&j, "lines").as_f64(), Some(2.0));
    assert_eq!(field(&j, "binary"), Json::Bool(false));
    let (_, j) = get("/api/fs/preview", &[("path", p), ("bytes", "5")]);
    assert_eq!(field(&j, "text").as_str(), Some("hello"));
    assert_eq!(field(&j, "truncated"), Json::Bool(true));
    for bad in ["0", "-1", "x", ""] {
        assert_eq!(
            get("/api/fs/preview", &[("path", p), ("bytes", bad)]).0,
            400,
            "{bad:?}"
        );
    }
    let (s, j) = get("/api/fs/preview", &[("path", d.to_str().unwrap())]);
    assert_eq!(s, 400);
    assert!(field(&j, "error").as_str().unwrap().contains("not a file"));
}

#[test]
fn other_methods_and_unknown_routes_are_refused() {
    assert_eq!(call("POST", "/api/fs/list", &[]).0, 405);
    assert_eq!(call("DELETE", "/api/fs/stat", &[]).0, 405);
    assert_eq!(get("/api/fs/nope", &[]).0, 404);
}

#[test]
fn the_api_router_sends_fs_paths_here() {
    use crate::app::{App, DeviceChoice};
    use crate::engine::CpuBackend;
    let a = std::sync::Arc::new(App::with_backend(
        std::sync::Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let req = Request {
        method: "GET".into(),
        path: "/api/fs/list".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &req).status, 200);
}
