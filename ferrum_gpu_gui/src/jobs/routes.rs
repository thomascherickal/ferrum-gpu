//! The HTTP routes of the jobs API:
//!
//! | route | does |
//! |---|---|
//! | `GET /api/jobs` | every remembered job, one summary each |
//! | `POST /api/jobs` `{"cmd":"train","args":["--data","t.txt",...]}` | start one; `{"id":N}` |
//! | `GET /api/jobs/N?since=K` | state, exit status, output lines from index K, last progress |
//! | `POST /api/jobs/N/stop` | ask it to stop; training and generation end cleanly at the next step or token |

use super::view;
use crate::app::App;
use crate::http::{Request, Response};
use crate::json::Json;

/// True if `path` belongs to the jobs API.
pub fn handles(path: &str) -> bool {
    path == "/api/jobs" || path.starts_with("/api/jobs/")
}

fn parse_body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    if text.trim().is_empty() {
        return Ok(Json::obj([]));
    }
    Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))
}

fn start(app: &App, req: &Request) -> Response {
    let j = match parse_body(req) {
        Ok(j) => j,
        Err(r) => return r,
    };
    let Some(cmd) = j.get("cmd").and_then(Json::as_str) else {
        return Response::error(400, "cmd is required");
    };
    let args = match j.get("args") {
        None | Some(Json::Null) => vec![],
        Some(Json::Arr(a)) => {
            let mut out = vec![];
            for v in a {
                match v.as_str() {
                    Some(s) => out.push(s.to_string()),
                    None => return Response::error(400, "args must be a list of strings"),
                }
            }
            out
        }
        Some(_) => return Response::error(400, "args must be a list of strings"),
    };
    match app.jobs.start(cmd, args) {
        Ok(job) => Response::json(200, &Json::obj([("id", Json::Num(job.id as f64))])),
        Err(e) => Response::error(400, &e),
    }
}

/// `/api/jobs/N` or `/api/jobs/N/stop` into `(N, rest)`.
fn split_id(path: &str) -> Option<(u64, &str)> {
    let tail = path.strip_prefix("/api/jobs/")?;
    let (id, rest) = tail.split_once('/').unwrap_or((tail, ""));
    Some((id.parse().ok()?, rest))
}

/// Answer a request for the jobs API (see [`handles`]).
pub fn route(app: &App, req: &Request) -> Response {
    if req.path == "/api/jobs" {
        return match req.method.as_str() {
            "GET" => Response::json(
                200,
                &Json::obj([(
                    "jobs",
                    Json::Arr(app.jobs.list().iter().map(|j| view::summary(j)).collect()),
                )]),
            ),
            "POST" => start(app, req),
            _ => Response::error(405, "method not allowed"),
        };
    }
    let Some((id, rest)) = split_id(&req.path) else {
        return Response::error(404, "no such job");
    };
    let Some(job) = app.jobs.get(id) else {
        return Response::error(404, "no such job");
    };
    match (req.method.as_str(), rest) {
        ("GET", "") => {
            let since = match req.param("since") {
                None => 0,
                Some(s) => match s.parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => return Response::error(400, "since must be a number"),
                },
            };
            Response::json(200, &view::poll(&job, since))
        }
        ("POST", "stop") => {
            // the body is optional and ignored (older pages sent {"force": bool}); it must still be JSON
            if let Err(r) = parse_body(req) {
                return r;
            }
            let asked = job.stop();
            Response::json(200, &Json::obj([("stopping", Json::Bool(asked))]))
        }
        (_, "" | "stop") => Response::error(405, "method not allowed"),
        _ => Response::error(404, "no such job route"),
    }
}
