//! The Models tab's API: look inside a GGUF file, and list the GGUF files in a folder. The work is in
//! `ferrum_gpu_core::modelinfo`; this module adds only the routes.
//!
//! | route | answers |
//! |---|---|
//! | `GET /api/gguf/inspect?path=P` | header, metadata, tensors and size of the GGUF file P (only its start is read) |
//! | `GET /api/gguf/models?dir=D` | the `.gguf` files under folder D (default: where the server started) |

use crate::http::{Request, Response};
use crate::json::Json;
use ferrum_gpu_core::modelinfo::{inspect, scan::scan, to_json};
use std::path::PathBuf;

/// True if `path` belongs to this API.
pub fn handles(path: &str) -> bool {
    path.starts_with("/api/gguf/")
}

fn clean(v: Option<&str>) -> Option<&str> {
    v.filter(|s| !s.is_empty() && !s.contains('\0'))
}

/// Answer a request for this API (see [`handles`]).
pub fn route(req: &Request) -> Response {
    if req.method != "GET" {
        return Response::error(405, "method not allowed");
    }
    match req.path.as_str() {
        "/api/gguf/inspect" => match clean(req.param("path")) {
            None => Response::error(400, "path is required"),
            Some(p) => match inspect(std::path::Path::new(p)) {
                Ok(i) => Response::json(200, &to_json(&i)),
                Err(e) => Response::error(400, &e),
            },
        },
        "/api/gguf/models" => {
            let dir = clean(req.param("dir"))
                .map(PathBuf::from)
                .unwrap_or_else(ferrum_gpu_core::places::cwd_or_dot);
            match scan(&dir) {
                Ok((files, truncated)) => Response::json(
                    200,
                    &Json::obj([
                        ("dir", Json::Str(dir.to_string_lossy().into_owned())),
                        (
                            "files",
                            Json::Arr(
                                files
                                    .iter()
                                    .map(|f| {
                                        Json::obj([
                                            (
                                                "path",
                                                Json::Str(f.path.to_string_lossy().into_owned()),
                                            ),
                                            (
                                                "name",
                                                Json::Str(
                                                    f.path
                                                        .file_name()
                                                        .map(|n| n.to_string_lossy().into_owned())
                                                        .unwrap_or_default(),
                                                ),
                                            ),
                                            ("size", Json::Num(f.size as f64)),
                                            (
                                                "modified",
                                                f.modified
                                                    .map_or(Json::Null, |m| Json::Num(m as f64)),
                                            ),
                                        ])
                                    })
                                    .collect(),
                            ),
                        ),
                        ("truncated", Json::Bool(truncated)),
                    ]),
                ),
                Err(e) => Response::error(400, &e),
            }
        }
        _ => Response::error(404, "no such route"),
    }
}

#[cfg(test)]
mod tests;
