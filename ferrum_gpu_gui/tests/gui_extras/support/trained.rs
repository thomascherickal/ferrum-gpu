use super::*;

/// A trained app: 60 steps, no validation share.
pub fn trained() -> Arc<App> {
    let a = app_with_model();
    let (first, done) = train(&a, None);
    assert_eq!(first.0, 200);
    assert_eq!(field(&done, "state").as_str(), Some("done"), "{done}");
    a
}

/// Like [`call`], with query parameters (`?a=b`) and a raw reply body.
pub fn call_raw(
    app: &Arc<App>,
    method: &str,
    path: &str,
    query: &[(&str, &str)],
    body: &[u8],
) -> (u16, Vec<u8>) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: body.to_vec(),
    };
    let r = route(app, &req);
    (r.status, r.body)
}
