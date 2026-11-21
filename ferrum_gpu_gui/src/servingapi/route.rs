use super::*;

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
