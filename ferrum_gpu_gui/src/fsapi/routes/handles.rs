use super::*;

/// True if `path` belongs to the file API.
pub fn handles(path: &str) -> bool {
    path.starts_with("/api/fs/")
}

pub(super) fn num(n: impl Into<f64>) -> Json {
    Json::Num(n.into())
}

pub(super) fn text(p: &Path) -> Json {
    Json::Str(p.to_string_lossy().into_owned())
}

pub(super) fn places() -> Json {
    let cwd = std::env::current_dir().ok();
    let home = ferrum_gpu_core::places::home_dir();
    let mut v = vec![];
    for (name, p) in [
        ("Where the server started", cwd),
        ("Home", home),
        ("Temp", Some(std::env::temp_dir())),
    ] {
        if let Some(p) = p {
            v.push(Json::obj([
                ("name", Json::Str(name.into())),
                ("path", text(&p)),
            ]));
        }
    }
    Json::Arr(v)
}

pub(super) fn need<'a>(req: &'a Request, name: &str) -> Result<&'a str, Response> {
    match req.param(name) {
        Some(v) if !v.is_empty() && !v.contains('\0') => Ok(v),
        _ => Err(Response::error(400, &format!("{name} is required"))),
    }
}
