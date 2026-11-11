//! `/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood`, `/v1/loglikelihood_rolling`.
//! Request and response shapes follow ferrum-cpu's `ferrum-serve` so clients written for it work
//! unchanged:
//!
//! - `POST /v1/loglikelihood` `{"requests":[{"context","continuation"}]}` gives
//!   `{"results":[{"logprob","n_tokens","greedy"}]}`;
//! - `POST /v1/loglikelihood_rolling` `{"texts":[...]}` gives `{"results":[{"logprob","n_tokens"}]}`;
//! - `POST /v1/generate` `{"requests":[{"prompt","max_tokens","stop","temperature","top_k","top_p",
//!   "seed","repetition_penalty"}]}` gives `{"results":[{"text","n_tokens"}]}`.

use crate::decoder::{Decoder, GenReq};
use crate::limits::{Limits, DEFAULT_MAX_TOKENS, MAX_STOPS, MAX_STOP_BYTES};
use crate::score::{pair, rolling};
use ferrum_gpu_gui::http::{Request, Response};
use ferrum_gpu_gui::json::Json;

fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

fn bad(status: u16, m: impl AsRef<str>) -> Response {
    Response::error(status, m.as_ref())
}

fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body).map_err(|_| bad(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| bad(400, format!("bad JSON: {e}")))
}

/// Reject keys outside `allowed` (ferrum-cpu denies unknown fields).
fn only(j: &Json, allowed: &[&str], what: &str) -> Result<(), Response> {
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

fn list<'a>(j: &'a Json, key: &str) -> Result<&'a [Json], Response> {
    match j.get(key) {
        Some(Json::Arr(a)) => Ok(a),
        _ => Err(bad(400, format!("'{key}' must be an array"))),
    }
}

fn string<'a>(j: &'a Json, key: &str) -> Result<&'a str, Response> {
    j.get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| bad(400, format!("'{key}' must be a string")))
}

fn number(j: &Json, key: &str, default: f64) -> Result<f64, Response> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(|| bad(400, format!("'{key}' must be a number"))),
    }
}

/// Batch and length limits, checked before any model compute.
fn check(l: &Limits, n: usize, lens: impl Iterator<Item = usize>) -> Result<(), Response> {
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

fn info(d: &dyn Decoder) -> Response {
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
        ]),
    )
}

fn results(items: Vec<Json>) -> Response {
    Response::json(200, &Json::obj([("results", Json::Arr(items))]))
}

fn scored(s: crate::score::Score, with_greedy: bool) -> Json {
    let mut kv = vec![
        ("logprob", Json::Num(s.logprob)),
        ("n_tokens", num(s.n_tokens)),
    ];
    if with_greedy {
        kv.push(("greedy", Json::Bool(s.greedy)));
    }
    Json::obj_vec(kv)
}

fn loglikelihood(d: &dyn Decoder, l: &Limits, req: &Request) -> Result<Response, Response> {
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

fn loglikelihood_rolling(d: &dyn Decoder, l: &Limits, req: &Request) -> Result<Response, Response> {
    let j = body(req)?;
    only(&j, &["texts"], "body")?;
    let texts: Vec<&str> = list(&j, "texts")?
        .iter()
        .map(|t| {
            t.as_str()
                .ok_or_else(|| bad(400, "'texts' must hold strings"))
        })
        .collect::<Result<_, _>>()?;
    check(l, texts.len(), texts.iter().map(|t| t.chars().count()))?;
    let out = texts
        .iter()
        .map(|t| rolling(d, t).map(|s| scored(s, false)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| bad(422, e))?;
    Ok(results(out))
}

const GEN_FIELDS: [&str; 9] = [
    "prompt",
    "max_tokens",
    "stop",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "repetition_penalty",
    "chat",
];

/// Validate one generate request (ferrum-cpu's boundary checks).
fn gen_request(r: &Json, l: &Limits) -> Result<(String, GenReq), Response> {
    only(r, &GEN_FIELDS, "request")?;
    let prompt = string(r, "prompt")?.to_string();
    let max_tokens = number(r, "max_tokens", DEFAULT_MAX_TOKENS.min(l.max_tokens) as f64)?;
    if max_tokens < 1.0 || max_tokens > l.max_tokens as f64 {
        return Err(bad(
            400,
            format!(
                "max_tokens {max_tokens} must be in 1..={} (the max_tokens limit)",
                l.max_tokens
            ),
        ));
    }
    let temperature = number(r, "temperature", 0.0)?;
    if temperature < 0.0 {
        return Err(bad(
            400,
            format!("temperature must be finite and >= 0, got {temperature}"),
        ));
    }
    let top_p = number(r, "top_p", 1.0)?;
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(bad(400, format!("top_p must be in (0, 1], got {top_p}")));
    }
    let penalty = number(r, "repetition_penalty", f64::from(l.repetition_penalty))?;
    if penalty <= 0.0 {
        return Err(bad(
            400,
            format!("repetition_penalty must be a positive number, got {penalty}"),
        ));
    }
    let top_k = number(r, "top_k", 0.0)?;
    let seed = number(r, "seed", 0.0)?;
    if top_k < 0.0 || seed < 0.0 {
        return Err(bad(400, "top_k and seed must not be negative"));
    }
    if matches!(r.get("chat"), Some(Json::Bool(true))) {
        return Err(bad(
            400,
            "chat is not supported: this model has no chat template",
        ));
    }
    let stops: Vec<String> = match r.get("stop") {
        None | Some(Json::Null) => vec![],
        Some(Json::Arr(a)) => a
            .iter()
            .map(|s| {
                s.as_str()
                    .map(String::from)
                    .ok_or_else(|| bad(400, "'stop' must hold strings"))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(bad(400, "'stop' must be an array of strings")),
    };
    if stops.len() > MAX_STOPS {
        return Err(bad(
            400,
            format!(
                "{} stop strings exceed the limit of {MAX_STOPS}",
                stops.len()
            ),
        ));
    }
    if stops.iter().any(String::is_empty) {
        return Err(bad(400, "empty stop string"));
    }
    if let Some(s) = stops.iter().find(|s| s.len() > MAX_STOP_BYTES) {
        return Err(bad(
            400,
            format!(
                "a stop string of {} bytes exceeds the limit of {MAX_STOP_BYTES}",
                s.len()
            ),
        ));
    }
    Ok((
        prompt,
        GenReq {
            max_new: max_tokens as usize,
            temperature: temperature as f32,
            top_k: top_k as usize,
            top_p: top_p as f32,
            repetition_penalty: penalty as f32,
            seed: seed as u64,
            stops,
        },
    ))
}

fn generate(d: &dyn Decoder, l: &Limits, req: &Request) -> Result<Response, Response> {
    let j = body(req)?;
    only(&j, &["requests"], "body")?;
    let reqs = list(&j, "requests")?;
    check(
        l,
        reqs.len(),
        reqs.iter()
            .filter_map(|r| r.get("prompt").and_then(Json::as_str))
            .map(|p| p.chars().count()),
    )?;
    let parsed = reqs
        .iter()
        .map(|r| gen_request(r, l))
        .collect::<Result<Vec<_>, _>>()?;
    let out = parsed
        .iter()
        .map(|(p, g)| {
            d.complete(p, g)
                .map(|(text, n)| Json::obj([("text", Json::Str(text)), ("n_tokens", num(n))]))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| bad(422, e))?;
    Ok(results(out))
}

#[cfg(test)]
mod tests;
