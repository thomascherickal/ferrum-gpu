use super::*;

pub(super) fn export(app: &App, req: &Request) -> Response {
    let quant = match req.param("quant").unwrap_or("f32") {
        "f32" => Quant::F32,
        "q8_0" => Quant::Q8_0,
        "q4_0" => Quant::Q4_0,
        other => {
            return Response::error(400, &format!("unknown quant '{other}' (f32, q8_0, q4_0)"))
        }
    };
    match lock(&app.model).as_ref() {
        Some(m) => {
            let mut r = Response::new(200, "application/octet-stream", to_gguf(m, quant));
            r.headers.push((
                "Content-Disposition".into(),
                "attachment; filename=\"model.gguf\"".into(),
            ));
            r
        }
        None => Response::error(404, "there is no model to export"),
    }
}

pub(super) fn import(app: &App, req: &Request) -> Response {
    if lock(&app.job).state == JobState::Running {
        return Response::error(409, "stop the training run first");
    }
    let parsed = Gguf::parse(req.body.clone())
        .map_err(|e| e.to_string())
        .and_then(|g| from_gguf(&g).map_err(|e| e.to_string()));
    match parsed {
        Ok(m) if m.num_params() > app.max_params => Response::error(
            413,
            &format!(
                "the model has {} parameters; the GUI holds at most {}",
                m.num_params(),
                app.max_params
            ),
        ),
        Ok(m) => {
            let out = model_json(&m);
            app.set_model(m);
            Response::json(200, &out)
        }
        Err(e) => Response::error(400, &format!("not a usable llama/qwen2 GGUF: {e}")),
    }
}

/// Route one request.
pub fn route(app: &Arc<App>, req: &Request) -> Response {
    if crate::jobs::routes::handles(&req.path) {
        return crate::jobs::routes::route(app, req);
    }
    if crate::capableapi::handles(&req.path) {
        return crate::capableapi::route(app, req);
    }
    if crate::servingapi::handles(&req.path) {
        return crate::servingapi::route(app, req);
    }
    if crate::modelsapi::handles(&req.path) {
        return crate::modelsapi::route(req);
    }
    if crate::fsapi::routes::handles(&req.path) {
        return crate::fsapi::routes::route(req);
    }
    let (m, p) = (req.method.as_str(), req.path.as_str());
    match (m, p) {
        ("GET", "/api/info") => info(app),
        ("POST", "/api/model") => new_model(app, req),
        ("POST", "/api/train") => train(app, req),
        ("GET", "/api/train/status") => train_status(app),
        ("POST", "/api/train/stop") => Response::json(
            200,
            &Json::obj([("stopping", Json::Bool(app.stop_training()))]),
        ),
        ("POST", "/api/generate") => generate(app, req),
        ("GET", "/api/model.gguf") => export(app, req),
        ("PUT", "/api/model.gguf") => import(app, req),
        ("POST", "/api/selftest/result") => {
            *lock(&app.selftest) = Some(String::from_utf8_lossy(&req.body).into_owned());
            Response::json(200, &Json::obj([("ok", Json::Bool(true))]))
        }
        ("GET", "/api/selftest/vectors") => Response::json(200, &crate::selftest::vectors()),
        ("GET", "/api/selftest/bench-model.gguf") => match crate::selftest::bench_gguf(
            req.param("shape").unwrap_or("small"),
            req.param("quant").unwrap_or("f32"),
        ) {
            Ok(b) => Response::new(200, "application/octet-stream", b),
            Err(e) => Response::error(400, &e),
        },
        ("GET", "/api/selftest/result") => match lock(&app.selftest).clone() {
            Some(s) => Response::new(200, "application/json", s),
            None => Response::error(404, "no self-test result yet"),
        },
        ("GET", _) => assets::serve(p).unwrap_or_else(|| Response::error(404, "not found")),
        (_, p) if p.starts_with("/api/") => Response::error(405, "method not allowed"),
        _ => Response::error(405, "method not allowed"),
    }
}
