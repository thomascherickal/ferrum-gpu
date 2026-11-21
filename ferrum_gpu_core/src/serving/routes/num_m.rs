use super::*;

pub(super) fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

pub(super) fn bad(status: u16, m: impl AsRef<str>) -> Response {
    Response::error(status, m.as_ref())
}

pub(super) fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body).map_err(|_| bad(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| bad(400, format!("bad JSON: {e}")))
}

/// Reject keys outside `allowed` (ferrum-cpu denies unknown fields).
pub(super) fn only(j: &Json, allowed: &[&str], what: &str) -> Result<(), Response> {
    match j {
        Json::Obj(kv) => kv
            .iter()
            .find(|(k, _)| !allowed.contains(&k.as_str()))
            .map_or(Ok(()), |(k, _)| {
                Err(bad(400, format!("{what}: unknown field '{k}'")))
            }),
        _ => Err(bad(400, format!("{what} must be an object"))),
    }
}

pub(super) fn list<'a>(j: &'a Json, key: &str) -> Result<&'a [Json], Response> {
    match j.get(key) {
        Some(Json::Arr(a)) => Ok(a),
        _ => Err(bad(400, format!("'{key}' must be an array"))),
    }
}

pub(super) fn string<'a>(j: &'a Json, key: &str) -> Result<&'a str, Response> {
    j.get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| bad(400, format!("'{key}' must be a string")))
}

pub(super) fn number(j: &Json, key: &str, default: f64) -> Result<f64, Response> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(|| bad(400, format!("'{key}' must be a number"))),
    }
}

/// Batch and length limits, checked before any model compute.
pub(super) fn check(
    l: &Limits,
    n: usize,
    lens: impl Iterator<Item = usize>,
) -> Result<(), Response> {
    if n > l.max_batch {
        return Err(bad(
            400,
            format!("{n} requests exceed max_batch {}", l.max_batch),
        ));
    }
    match lens.max() {
        Some(c) if c > l.max_chars => Err(bad(
            400,
            format!("{c} chars exceed max_chars {}", l.max_chars),
        )),
        _ => Ok(()),
    }
}

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

pub(super) fn info(d: &dyn Decoder) -> Response {
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

pub(super) fn results(items: Vec<Json>) -> Response {
    Response::json(200, &Json::obj([("results", Json::Arr(items))]))
}

pub(super) fn scored(s: crate::serving::score::Score, with_greedy: bool) -> Json {
    let mut kv = vec![
        ("logprob", Json::Num(s.logprob)),
        ("n_tokens", num(s.n_tokens)),
    ];
    if with_greedy {
        kv.push(("greedy", Json::Bool(s.greedy)));
    }
    Json::obj_vec(kv)
}

pub(super) fn loglikelihood(
    d: &dyn Decoder,
    l: &Limits,
    req: &Request,
) -> Result<Response, Response> {
    let j = body(req)?;
    only(&j, &["requests"], "body")?;
    let reqs = list(&j, "requests")?;
    for r in reqs {
        only(r, &["context", "continuation"], "request")?;
    }
    let pairs: Vec<(&str, &str)> = reqs
        .iter()
        .map(|r| Ok((string(r, "context")?, string(r, "continuation")?)))
        .collect::<Result<_, Response>>()?;
    check(
        l,
        pairs.len(),
        pairs
            .iter()
            .map(|(c, k)| c.chars().count() + k.chars().count()),
    )?;
    let out = pairs
        .iter()
        .map(|(c, k)| pair(d, c, k).map(|s| scored(s, true)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| bad(422, e))?;
    Ok(results(out))
}
