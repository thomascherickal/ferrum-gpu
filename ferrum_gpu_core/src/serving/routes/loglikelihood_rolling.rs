use super::*;

pub(crate) fn loglikelihood_rolling(
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

pub(crate) const GEN_FIELDS: [&str; 9] = [
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
