use super::*;

pub(super) fn loglikelihood_rolling(
    d: &dyn Decoder,
    l: &Limits,
    req: &Request,
) -> Result<Response, Response> {
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

pub(super) const GEN_FIELDS: [&str; 9] = [
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
pub(super) fn gen_request(r: &Json, l: &Limits) -> Result<(String, GenReq), Response> {
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

pub(super) fn generate(d: &dyn Decoder, l: &Limits, req: &Request) -> Result<Response, Response> {
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
