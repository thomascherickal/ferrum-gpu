use super::*;

/// True if `path` belongs to this API.
pub fn handles(path: &str) -> bool {
    path == "/api/capable" || path == "/api/system"
}

pub(super) fn number(req: &Request, name: &str) -> Result<Option<f64>, Response> {
    match req.param(name) {
        None | Some("") => Ok(None),
        Some(s) => s
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(Some)
            .ok_or_else(|| Response::error(400, &format!("{name} must be a number"))),
    }
}

pub(super) fn capable(req: &Request) -> Response {
    let memory = match number(req, "memory_gb") {
        Ok(Some(m)) if m > 0.0 && m <= 1.0e6 => m,
        Ok(_) => return Response::error(400, "memory_gb must be above 0 and at most 1000000"),
        Err(r) => return r,
    };
    let unified = matches!(req.param("unified"), Some("1" | "true"));
    let mut out = vec![("row".to_string(), row_json(&row(memory, unified)))];
    let (bw, model) = match (number(req, "bandwidth"), number(req, "model_b")) {
        (Ok(b), Ok(m)) => (b, m),
        (Err(r), _) | (_, Err(r)) => return r,
    };
    if let (Some(bw), Some(m)) = (bw, model) {
        if bw <= 0.0 || m <= 0.0 {
            return Response::error(400, "bandwidth and model_b must be above 0");
        }
        let params = m * 1e9;
        out.push((
            "ceiling".to_string(),
            Json::Arr(
                Precision::ALL
                    .iter()
                    .map(|p| {
                        Json::obj([
                            ("precision", Json::Str(p.name().into())),
                            (
                                "tokens_per_s",
                                Json::Num(decode_ceiling_tps(bw, params, *p)),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ));
    }
    Response::json(200, &Json::Obj(out))
}
