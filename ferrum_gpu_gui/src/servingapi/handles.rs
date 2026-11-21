use super::*;

/// True if `path` belongs to this API.
pub fn handles(path: &str) -> bool {
    path.starts_with("/api/serve/") || path == "/api/tools"
}

pub(super) fn status_json(s: &Status) -> Json {
    Json::obj([
        ("running", Json::Bool(s.running)),
        ("addr", s.addr.clone().map_or(Json::Null, Json::Str)),
        ("has_token", Json::Bool(s.has_token)),
        ("seconds", s.seconds.map_or(Json::Null, Json::Num)),
        (
            "lines",
            Json::Arr(s.lines.iter().cloned().map(Json::Str).collect()),
        ),
    ])
}

pub(super) fn tool(name: &str, env: &str, fallback: &str) -> Json {
    tool_json(name, find_tool(None, env, name, fallback))
}

/// How the page is told about a tool: where it is, or why it was not found.
pub(super) fn tool_json(name: &str, found: Result<std::path::PathBuf, String>) -> Json {
    let (path, note) = match found {
        Ok(p) => (Json::Str(p.to_string_lossy().into_owned()), Json::Null),
        Err(e) => (Json::Null, Json::Str(e)),
    };
    Json::obj([
        ("name", Json::Str(name.into())),
        ("path", path),
        ("note", note),
    ])
}

pub(super) fn start_args(req: &Request) -> Result<Vec<String>, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    let j = Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))?;
    match j.get("args") {
        Some(Json::Arr(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| Response::error(400, "args must be a list of strings"))
            })
            .collect(),
        _ => Err(Response::error(400, "args must be a list of strings")),
    }
}
