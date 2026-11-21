use super::*;

/// A number, or `null` when absent.
pub(crate) fn opt_num(v: Option<f32>) -> Json {
    v.map_or(Json::Null, |x| Json::Num(f64::from(x)))
}

pub(crate) fn generate(app: &App, req: &Request) -> Response {
    let j = match body(req) {
        Ok(j) => j,
        Err(r) => return r,
    };
    let Some(model) = lock(&app.model).clone() else {
        return Response::error(400, "create or load a model first");
    };
    if model.cfg.vocab != tokenizer::VOCAB {
        return Response::error(
            400,
            "this model's vocabulary is not bytes; text in and out needs a byte-level model",
        );
    }
    let prompt = match j.get("prompt").and_then(Json::as_str) {
        Some(p) if !p.is_empty() && p.len() <= MAX_PROMPT => p,
        Some("") => return Response::error(400, "the prompt is empty"),
        Some(_) => return Response::error(413, &format!("the prompt is over {MAX_PROMPT} bytes")),
        None => return Response::error(400, "'prompt' (a string) is required"),
    };
    let opts = (|| {
        Ok::<_, Response>(GenOpts {
            max_new: field(&j, "max_new", 64.0, 1.0, MAX_NEW as f64)? as usize,
            temperature: field(&j, "temperature", 0.0, 0.0, 10.0)? as f32,
            top_k: field(&j, "top_k", 0.0, 0.0, 256.0)? as usize,
            seed: field(&j, "seed", 1.0, 0.0, 9.0e15)? as u64,
        })
    })();
    let opts = match opts {
        Ok(o) => o,
        Err(r) => return r,
    };
    let stops = match stop::parse(&j) {
        Ok(s) => s,
        Err(e) => return Response::error(400, &e),
    };
    match app
        .backend
        .generate(&model, &tokenizer::encode(prompt), &opts)
    {
        Ok(ids) => Response::json(
            200,
            &Json::obj([
                (
                    "text",
                    Json::Str(stop::cut_at_stop(
                        tokenizer::decode(&ids).unwrap_or_default(),
                        &stops,
                    )),
                ),
                ("tokens", Json::Arr(ids.iter().map(|&t| num(t)).collect())),
                ("device", Json::Str(app.backend.kind().into())),
            ]),
        ),
        Err(e) => Response::error(400, &e),
    }
}
