//! `/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood`.

use crate::decoder::Decoder;
use ferrum_gpu_gui::engine::GenOpts;
use ferrum_gpu_gui::http::{Request, Response};
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::tokenizer::{decode, encode};

/// Largest prompt or context accepted (bytes).
pub const MAX_PROMPT: usize = 4096;
/// Most tokens one generate call may add.
pub const MAX_NEW: usize = 1024;
/// Largest request body (bytes).
pub const MAX_BODY: usize = 1 << 20;

fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))
}

fn text<'a>(j: &'a Json, key: &str) -> Result<&'a str, Response> {
    j.get(key)
        .and_then(Json::as_str)
        .filter(|s| s.len() <= MAX_PROMPT)
        .ok_or_else(|| {
            Response::error(
                400,
                &format!("'{key}' must be a string of at most {MAX_PROMPT} bytes"),
            )
        })
}

fn field(j: &Json, key: &str, default: f64, lo: f64, hi: f64) -> Result<f64, Response> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(v) => match v.as_f64() {
            Some(n) if n >= lo && n <= hi => Ok(n),
            _ => Err(Response::error(
                400,
                &format!("'{key}' must be a number in {lo}..={hi}"),
            )),
        },
    }
}

/// Route one request.
pub fn route(d: &dyn Decoder, req: &Request) -> Response {
    let result = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => Ok(Response::json(
            200,
            &Json::obj([("status", Json::Str("ok".into()))]),
        )),
        ("GET", "/v1/info") => Ok(info(d)),
        ("POST", "/v1/generate") => generate(d, req),
        ("POST", "/v1/loglikelihood") => loglikelihood(d, req),
        (_, "/health" | "/v1/info" | "/v1/generate" | "/v1/loglikelihood") => {
            Ok(Response::error(405, "method not allowed"))
        }
        _ => Ok(Response::error(404, "not found")),
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
            ("params", num(i.params)),
            ("vocab", num(i.vocab)),
            ("ctx", num(i.ctx)),
        ]),
    )
}

fn generate(d: &dyn Decoder, req: &Request) -> Result<Response, Response> {
    let j = body(req)?;
    let prompt = text(&j, "prompt")?;
    let opts = GenOpts {
        max_new: field(&j, "max_new", 64.0, 1.0, MAX_NEW as f64)? as usize,
        temperature: field(&j, "temperature", 0.0, 0.0, 10.0)? as f32,
        top_k: field(&j, "top_k", 0.0, 0.0, 1_000_000.0)? as usize,
        seed: field(&j, "seed", 1.0, 0.0, 9.0e15)? as u64,
    };
    let new = d
        .generate(&encode(prompt), &opts)
        .map_err(|e| Response::error(400, &e))?;
    let out = decode(&new).map_err(|e| Response::error(500, &e))?;
    Ok(Response::json(
        200,
        &Json::obj([("text", Json::Str(out)), ("tokens", num(new.len()))]),
    ))
}

fn loglikelihood(d: &dyn Decoder, req: &Request) -> Result<Response, Response> {
    let j = body(req)?;
    let (context, cont) = (text(&j, "context")?, text(&j, "continuation")?);
    if context.is_empty() || cont.is_empty() {
        return Err(Response::error(
            400,
            "'context' and 'continuation' must not be empty",
        ));
    }
    let ids = [encode(context), encode(cont)].concat();
    let scores = d.score(&ids).map_err(|e| Response::error(400, &e))?;
    let tail = &scores[scores.len() - encode(cont).len()..];
    let total: f64 = tail.iter().map(|(lp, _)| f64::from(*lp)).sum();
    Ok(Response::json(
        200,
        &Json::obj([
            ("loglikelihood", Json::Num(total)),
            ("is_greedy", Json::Bool(tail.iter().all(|(_, g)| *g))),
            ("tokens", num(tail.len())),
        ]),
    ))
}

#[cfg(test)]
mod tests;
