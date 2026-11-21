use super::*;

/// Answer a request for this API (see [`handles`]).
pub fn route(app: &App, req: &Request) -> Response {
    if req.method != "GET" {
        return Response::error(405, "method not allowed");
    }
    match req.path.as_str() {
        "/api/capable" => capable(req),
        "/api/system" => system(app),
        _ => Response::error(404, "no such route"),
    }
}
