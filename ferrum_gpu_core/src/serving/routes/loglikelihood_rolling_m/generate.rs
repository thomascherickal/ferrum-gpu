use super::*;

pub(crate) fn generate(d: &dyn Decoder, l: &Limits, req: &Request) -> Result<Response, Response> {
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
