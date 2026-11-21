use super::*;

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
