//! The HTTP routes of the file API (all `GET`):
//!
//! | route | answers |
//! |---|---|
//! | `/api/fs/list?dir=P` | the entries of folder P (default: where the server was started), and the places to start from |
//! | `/api/fs/stat?path=P` | `file`, `dir` or `missing`, with file count and bytes |
//! | `/api/fs/preview?path=P&bytes=N` | the first N bytes of a file as text, its size and line count |

use super::{list, preview, stat, MAX_PREVIEW};
use crate::http::{Request, Response};
use crate::json::Json;
use std::path::{Path, PathBuf};

/// True if `path` belongs to the file API.
pub fn handles(path: &str) -> bool {
    path.starts_with("/api/fs/")
}

fn num(n: impl Into<f64>) -> Json {
    Json::Num(n.into())
}

fn text(p: &Path) -> Json {
    Json::Str(p.to_string_lossy().into_owned())
}

fn places() -> Json {
    let cwd = std::env::current_dir().ok();
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
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

fn need<'a>(req: &'a Request, name: &str) -> Result<&'a str, Response> {
    match req.param(name) {
        Some(v) if !v.is_empty() && !v.contains('\0') => Ok(v),
        _ => Err(Response::error(400, &format!("{name} is required"))),
    }
}

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
