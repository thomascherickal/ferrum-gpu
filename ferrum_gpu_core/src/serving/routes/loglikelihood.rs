use super::*;

pub(crate) fn loglikelihood(
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
