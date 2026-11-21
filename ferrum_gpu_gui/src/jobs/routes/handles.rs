use super::*;

/// True if `path` belongs to the jobs API.
pub fn handles(path: &str) -> bool {
    path == "/api/jobs" || path.starts_with("/api/jobs/")
}

pub(super) fn parse_body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    if text.trim().is_empty() {
        return Ok(Json::obj([]));
    }
    Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))
}

pub(super) fn start(app: &App, req: &Request) -> Response {
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
pub(super) fn split_id(path: &str) -> Option<(u64, &str)> {
    let tail = path.strip_prefix("/api/jobs/")?;
    let (id, rest) = tail.split_once('/').unwrap_or((tail, ""));
    Some((id.parse().ok()?, rest))
}
