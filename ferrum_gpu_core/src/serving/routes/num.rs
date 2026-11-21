use super::*;

pub(crate) fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

pub(crate) fn bad(status: u16, m: impl AsRef<str>) -> Response {
    Response::error(status, m.as_ref())
}

pub(crate) fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body).map_err(|_| bad(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| bad(400, format!("bad JSON: {e}")))
}

/// Reject keys outside `allowed` (ferrum-cpu denies unknown fields).
pub(crate) fn only(j: &Json, allowed: &[&str], what: &str) -> Result<(), Response> {
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

pub(crate) fn list<'a>(j: &'a Json, key: &str) -> Result<&'a [Json], Response> {
    match j.get(key) {
        Some(Json::Arr(a)) => Ok(a),
        _ => Err(bad(400, format!("'{key}' must be an array"))),
    }
}

pub(crate) fn string<'a>(j: &'a Json, key: &str) -> Result<&'a str, Response> {
    j.get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| bad(400, format!("'{key}' must be a string")))
}

pub(crate) fn number(j: &Json, key: &str, default: f64) -> Result<f64, Response> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(|| bad(400, format!("'{key}' must be a number"))),
    }
}

/// Batch and length limits, checked before any model compute.
pub(crate) fn check(
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
