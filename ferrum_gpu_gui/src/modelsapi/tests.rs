use super::*;
use ferrum_gguf::{GgufBuilder, Value};
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ferrum_modelsapi_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn model(dir: &std::path::Path, rel: &str) -> PathBuf {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.tensor_f32("w", &[4], &[1.0; 4]);
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, b.into_bytes()).unwrap();
    p
}

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

#[test]
fn the_api_router_sends_gguf_paths_here() {
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
        path: "/api/gguf/models".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &req).status, 200);
}
