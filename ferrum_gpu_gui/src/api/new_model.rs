use super::*;

pub(crate) fn new_model(app: &App, req: &Request) -> Response {
    let j = match body(req) {
        Ok(j) => j,
        Err(r) => return r,
    };
    if lock(&app.job).state == JobState::Running {
        return Response::error(409, "stop the training run first");
    }
    let name = j.get("preset").and_then(Json::as_str).unwrap_or("tiny");
    let seed = match field(&j, "seed", 1.0, 0.0, 9.0e15) {
        Ok(s) => s as u64,
        Err(r) => return r,
    };
    let tied = match j.get("tied") {
        None | Some(Json::Null) => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => return Response::error(400, "tied must be true or false"),
    };
    match fresh_model_with(name, seed, tied) {
        Ok(m) => {
            let out = model_json(&m);
            app.set_model(m);
            Response::json(200, &out)
        }
        Err(e) => Response::error(400, &e),
    }
}

pub(crate) fn train(app: &Arc<App>, req: &Request) -> Response {
    let j = match body(req) {
        Ok(j) => j,
        Err(r) => return r,
    };
    let text = match j.get("text").and_then(Json::as_str) {
        Some(t) if t.len() <= MAX_TEXT => t.as_bytes().to_vec(),
        Some(_) => return Response::error(413, &format!("the text is over {MAX_TEXT} bytes")),
        None => return Response::error(400, "'text' (a string) is required"),
    };
    let parsed = (|| {
        Ok::<_, Response>(TrainSpec {
            text,
            steps: field(&j, "steps", 100.0, 1.0, 100_000.0)? as usize,
            lr: field(&j, "lr", 0.01, 1e-6, 1.0)? as f32,
            batch: field(&j, "batch", 8.0, 1.0, 256.0)? as usize,
            context: field(&j, "context", 32.0, 1.0, 4096.0)? as usize,
            seed: field(&j, "seed", 1.0, 0.0, 9.0e15)? as u64,
        })
    })();
    let val = match field(&j, "val", 0.0, 0.0, f64::from(validation::MAX_FRACTION)) {
        Ok(v) => v as f32,
        Err(r) => return r,
    };
    match parsed.and_then(|s| {
        app.start_training_validated(s, val)
            .map_err(|e| Response::error(if e.contains("already") { 409 } else { 400 }, &e))
    }) {
        Ok(()) => Response::json(200, &Json::obj([("state", Json::Str("running".into()))])),
        Err(r) => r,
    }
}

pub(super) fn train_status(app: &App) -> Response {
    let job = lock(&app.job);
    let p = lock(&job.progress).clone();
    let error = match &job.state {
        JobState::Failed(e) => Json::Str(e.clone()),
        _ => Json::Null,
    };
    Response::json(
        200,
        &Json::obj([
            ("state", Json::Str(job.state.word().into())),
            ("step", num(p.step)),
            ("total", num(p.total)),
            (
                "losses",
                Json::Arr(p.losses.iter().map(|&l| Json::Num(f64::from(l))).collect()),
            ),
            ("error", error),
            ("val_before", opt_num(p.val_before)),
            ("val_after", opt_num(p.val_after)),
        ]),
    )
}
