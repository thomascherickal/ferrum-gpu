//! The Serve and Export tabs' API: run the JSON server in this process and stop it, and say which
//! export tools (`ollama`, `lms`) are installed. The server and its startup are in
//! `ferrum_gpu_core::serving`; the tool lookup is the one the export commands use.
//!
//! | route | does |
//! |---|---|
//! | `GET /api/serve/status` | whether a server runs, its address, whether it needs a token, and its status lines |
//! | `POST /api/serve/start` `{"args":["--model","m.gguf","--port","8080"]}` | start it with the `ferrum-gpu-server` flags |
//! | `POST /api/serve/stop` | stop it |
//! | `GET /api/tools` | `ollama` and `lms`: where each is found, or why not |

use crate::app::App;
use crate::http::{Request, Response};
use crate::json::Json;
use ferrum_gpu_core::commands::runtimes::find_tool;
use ferrum_gpu_core::serving::host::Status;

/// True if `path` belongs to this API.
pub fn handles(path: &str) -> bool {
    path.starts_with("/api/serve/") || path == "/api/tools"
}

fn status_json(s: &Status) -> Json {
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

fn tool(name: &str, env: &str, fallback: &str) -> Json {
    let (path, note) = match find_tool(None, env, name, fallback) {
        Ok(p) => (Json::Str(p.to_string_lossy().into_owned()), Json::Null),
        Err(e) => (Json::Null, Json::Str(e)),
    };
    Json::obj([
        ("name", Json::Str(name.into())),
        ("path", path),
        ("note", note),
    ])
}

fn start_args(req: &Request) -> Result<Vec<String>, Response> {
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

/// Answer a request for this API (see [`handles`]).
pub fn route(app: &App, req: &Request) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/api/serve/status") => Response::json(200, &status_json(&app.serve.status())),
        ("POST", "/api/serve/start") => {
            let args = match start_args(req) {
                Ok(a) => a,
                Err(r) => return r,
            };
            if args.iter().any(|a| a.contains('\0')) {
                return Response::error(400, "an argument contains a NUL byte");
            }
            match app.serve.start(&args) {
                Ok(s) => Response::json(200, &status_json(&s)),
                Err(e) => Response::error(400, &e),
            }
        }
        ("POST", "/api/serve/stop") => {
            Response::json(200, &Json::obj([("stopped", Json::Bool(app.serve.stop()))]))
        }
        ("GET", "/api/tools") => Response::json(
            200,
            &Json::obj([(
                "tools",
                Json::Arr(vec![
                    tool("ollama", "FERRUM_OLLAMA", ".local/bin/ollama"),
                    tool("lms", "FERRUM_LMS", ".lmstudio/bin/lms"),
                ]),
            )]),
        ),
        (_, "/api/serve/status" | "/api/serve/start" | "/api/serve/stop" | "/api/tools") => {
            Response::error(405, "method not allowed")
        }
        _ => Response::error(404, "no such route"),
    }
}

#[cfg(test)]
mod tests;
