use super::*;

/// Answer a request for the file API (see [`handles`]).
pub fn route(req: &Request) -> Response {
    if req.method != "GET" {
        return Response::error(405, "method not allowed");
    }
    match req.path.as_str() {
        "/api/fs/list" => {
            let dir = match req.param("dir") {
                Some(d) if !d.is_empty() => PathBuf::from(d),
                _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            };
            if dir.to_string_lossy().contains('\0') {
                return Response::error(400, "dir contains a NUL byte");
            }
            match list::list_dir(&dir) {
                Ok(l) => Response::json(
                    200,
                    &Json::obj([
                        ("dir", text(&l.dir)),
                        ("parent", l.parent.as_deref().map_or(Json::Null, text)),
                        (
                            "entries",
                            Json::Arr(
                                l.entries
                                    .iter()
                                    .map(|e| {
                                        Json::obj([
                                            ("name", Json::Str(e.name.clone())),
                                            ("kind", Json::Str(e.kind.word().into())),
                                            ("size", num(e.size as f64)),
                                            ("hidden", Json::Bool(e.hidden)),
                                        ])
                                    })
                                    .collect(),
                            ),
                        ),
                        ("truncated", Json::Bool(l.truncated)),
                        ("places", places()),
                    ]),
                ),
                Err(e) => Response::error(400, &e),
            }
        }
        "/api/fs/stat" => match need(req, "path") {
            Ok(p) => {
                let s = stat::stat(Path::new(p));
                Response::json(
                    200,
                    &Json::obj([
                        ("kind", Json::Str(s.kind.into())),
                        ("bytes", num(s.bytes as f64)),
                        ("files", num(s.files as f64)),
                        ("truncated", Json::Bool(s.truncated)),
                    ]),
                )
            }
            Err(r) => r,
        },
        "/api/fs/preview" => {
            let path = match need(req, "path") {
                Ok(p) => p,
                Err(r) => return r,
            };
            let bytes = match req.param("bytes") {
                None => 2_000,
                Some(b) => match b.parse::<usize>() {
                    Ok(n) if n >= 1 => n.min(MAX_PREVIEW),
                    _ => return Response::error(400, "bytes must be a positive number"),
                },
            };
            match preview::preview(Path::new(path), bytes) {
                Ok(p) => Response::json(
                    200,
                    &Json::obj([
                        ("text", Json::Str(p.text)),
                        ("size", num(p.size as f64)),
                        ("lines", num(p.lines as f64)),
                        ("truncated", Json::Bool(p.truncated)),
                        ("binary", Json::Bool(p.binary)),
                    ]),
                ),
                Err(e) => Response::error(400, &e),
            }
        }
        _ => Response::error(404, "no such route"),
    }
}
