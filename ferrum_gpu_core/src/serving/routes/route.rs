use super::*;

/// Route one request.
pub fn route(d: &dyn Decoder, l: &Limits, req: &Request) -> Response {
    let result = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => Ok(Response::json(
            200,
            &Json::obj([("status", Json::Str("ok".into()))]),
        )),
        ("GET", "/v1/info") => Ok(info(d)),
        ("POST", "/v1/generate") => generate(d, l, req),
        ("POST", "/v1/loglikelihood") => loglikelihood(d, l, req),
        ("POST", "/v1/loglikelihood_rolling") => loglikelihood_rolling(d, l, req),
        (
            _,
            "/health"
            | "/v1/info"
            | "/v1/generate"
            | "/v1/loglikelihood"
            | "/v1/loglikelihood_rolling",
        ) => Ok(bad(405, "method not allowed")),
        _ => Ok(bad(404, "not found")),
    };
    result.unwrap_or_else(|r| r)
}

pub(crate) fn info(d: &dyn Decoder) -> Response {
    let i = d.info();
    Response::json(
        200,
        &Json::obj([
            ("device", Json::Str(i.device)),
            ("kind", Json::Str(i.kind)),
            ("arch", Json::Str(i.arch)),
            ("params", num(i.params)),
            ("vocab", num(i.vocab)),
            ("ctx", num(i.ctx)),
            // ferrum-cpu's name for the same number; its LightEval adapter reads this one
            ("context_len", num(i.ctx)),
        ]),
    )
}

pub(crate) fn results(items: Vec<Json>) -> Response {
    Response::json(200, &Json::obj([("results", Json::Arr(items))]))
}

pub(crate) fn scored(s: crate::serving::score::Score, with_greedy: bool) -> Json {
    let mut kv = vec![
        ("logprob", Json::Num(s.logprob)),
        ("n_tokens", num(s.n_tokens)),
    ];
    if with_greedy {
        kv.push(("greedy", Json::Bool(s.greedy)));
    }
    Json::obj_vec(kv)
}
