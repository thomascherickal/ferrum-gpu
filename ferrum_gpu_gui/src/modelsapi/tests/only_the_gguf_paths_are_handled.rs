use super::*;

#[test]
fn only_the_gguf_paths_are_handled() {
    assert!(handles("/api/gguf/inspect") && handles("/api/gguf/models"));
    assert!(!handles("/api/gguf") && !handles("/api/fs/list") && !handles("/api/jobs"));
}

#[test]
fn inspect_describes_a_model_and_refuses_everything_else() {
    let d = tmp();
    let p = model(&d, "m.gguf");
    let (s, j) = get("/api/gguf/inspect", &[("path", p.to_str().unwrap())]);
    assert_eq!(s, 200, "{j}");
    assert_eq!(field(&j, "architecture").as_str(), Some("gpt2"));
    assert_eq!(field(&j, "n_tensors").as_f64(), Some(1.0));
    assert_eq!(field(&j, "parameters").as_f64(), Some(4.0));
    for q in [vec![], vec![("path", "")], vec![("path", "a\0b")]] {
        let (s, j) = get("/api/gguf/inspect", &q);
        assert_eq!(s, 400);
        assert!(field(&j, "error")
            .as_str()
            .unwrap()
            .contains("path is required"));
    }
    let txt = d.join("t.gguf");
    std::fs::write(&txt, b"not gguf").unwrap();
    let (s, j) = get("/api/gguf/inspect", &[("path", txt.to_str().unwrap())]);
    assert_eq!(s, 400);
    assert!(field(&j, "error").as_str().unwrap().contains("magic"));
    assert_eq!(
        get("/api/gguf/inspect", &[("path", d.to_str().unwrap())]).0,
        400
    );
    assert_eq!(
        get("/api/gguf/inspect", &[("path", "/no/such/model.gguf")]).0,
        400
    );
}

#[test]
fn the_model_list_has_name_path_size_and_time_for_each_file() {
    let d = tmp();
    let a = model(&d, "a.gguf");
    model(&d, "sub/b.gguf");
    std::fs::write(d.join("readme.txt"), b"x").unwrap();
    let (s, j) = get("/api/gguf/models", &[("dir", d.to_str().unwrap())]);
    assert_eq!(s, 200, "{j}");
    assert_eq!(field(&j, "dir").as_str(), d.to_str());
    assert_eq!(field(&j, "truncated"), Json::Bool(false));
    let Json::Arr(files) = field(&j, "files") else {
        panic!("{j}")
    };
    assert_eq!(files.len(), 2);
    assert_eq!(field(&files[0], "name").as_str(), Some("a.gguf"));
    assert_eq!(field(&files[0], "path").as_str(), a.to_str());
    assert_eq!(
        field(&files[0], "size").as_f64(),
        Some(std::fs::metadata(&a).unwrap().len() as f64)
    );
    assert!(field(&files[0], "modified").as_f64().is_some());
    assert_eq!(field(&files[1], "name").as_str(), Some("b.gguf"));
}

#[test]
fn the_default_folder_is_where_the_server_started_and_a_bad_one_is_a_400() {
    assert_eq!(get("/api/gguf/models", &[]).0, 200);
    assert_eq!(get("/api/gguf/models", &[("dir", "")]).0, 200);
    let (s, j) = get("/api/gguf/models", &[("dir", "/no/such/folder")]);
    assert_eq!(s, 400);
    assert!(field(&j, "error")
        .as_str()
        .unwrap()
        .contains("no/such/folder"));
}

#[test]
fn other_methods_and_unknown_routes_are_refused() {
    assert_eq!(call("POST", "/api/gguf/inspect", &[]).0, 405);
    assert_eq!(call("DELETE", "/api/gguf/models", &[]).0, 405);
    assert_eq!(get("/api/gguf/nope", &[]).0, 404);
}
