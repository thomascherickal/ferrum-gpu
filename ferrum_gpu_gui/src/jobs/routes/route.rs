use super::*;

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
