//! The JSON API and static routes.

use crate::app::{lock, App, JobState};
use crate::engine::{fresh_model, preset, GenOpts, TrainSpec};
use crate::http::{Request, Response};
use crate::json::Json;
use crate::{assets, tokenizer};
use ferrum_gguf::Gguf;
use ferrum_model::gguf::gpt2::Quant;
use ferrum_model::gguf::llama::{from_gguf, to_gguf};
use ferrum_ref::llama::Llama;
use std::sync::Arc;

/// Largest training text accepted (bytes).
pub const MAX_TEXT: usize = 1 << 20;
/// Largest prompt accepted (bytes).
pub const MAX_PROMPT: usize = 4096;
/// Most tokens one generate call may add.
pub const MAX_NEW: usize = 1024;
/// Largest model (parameters) the GUI will hold; bigger uploads are refused.
pub const MAX_PARAMS: usize = 60_000_000;
/// Largest request body the server reads (a GGUF upload).
pub const MAX_BODY: usize = 256 << 20;

fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))
}

/// An optional number field: absent gives `default`, wrong type or range is an error.
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

fn model_json(m: &Llama) -> Json {
    let c = &m.cfg;
    Json::obj([
        (
            "preset",
            App::preset_of(m).map_or(Json::Null, |p| Json::Str(p.into())),
        ),
        ("params", num(m.num_params())),
        ("vocab", num(c.vocab)),
        ("ctx", num(c.ctx)),
        ("dim", num(c.dim)),
        ("layers", num(c.layers)),
        ("heads", num(c.heads)),
        ("kv_heads", num(c.kv_heads)),
        ("hidden", num(c.hidden)),
        (
            "rope",
            Json::Str(format!("{:?}", c.rope_type).to_lowercase()),
        ),
        ("tied", Json::Bool(m.tied)),
        ("byte_tokens", Json::Bool(c.vocab == tokenizer::VOCAB)),
    ])
}

fn info(app: &App) -> Response {
    let adapters = app
        .adapters
        .iter()
        .map(|a| {
            Json::obj([
                ("name", Json::Str(a.name.clone())),
                ("backend", Json::Str(a.backend.clone())),
                ("kind", Json::Str(format!("{:?}", a.kind))),
                ("max_binding_mib", num((a.max_binding / (1 << 20)) as usize)),
            ])
        })
        .collect();
    let presets = ["tiny", "small", "medium"]
        .into_iter()
        .map(|n| {
            let c = preset(n).expect("built-in preset");
            let params = ferrum_ref::llama::Llama::random(c, 1, false, false).num_params();
            Json::obj([
                ("name", Json::Str(n.into())),
                ("params", num(params)),
                ("ctx", num(c.ctx)),
            ])
        })
        .collect();
    let model = lock(&app.model).as_ref().map_or(Json::Null, model_json);
    Response::json(
        200,
        &Json::obj([
            ("version", Json::Str(env!("CARGO_PKG_VERSION").into())),
            (
                "device",
                Json::obj([
                    ("requested", Json::Str(app.requested.as_str().into())),
                    ("active", Json::Str(app.backend.kind().into())),
                    ("name", Json::Str(app.backend.name())),
                    (
                        "note",
                        app.fallback_note.clone().map_or(Json::Null, Json::Str),
                    ),
                ]),
            ),
            ("adapters", Json::Arr(adapters)),
            ("presets", Json::Arr(presets)),
            ("model", model),
            (
                "limits",
                Json::obj([
                    ("max_text", num(MAX_TEXT)),
                    ("max_prompt", num(MAX_PROMPT)),
                    ("max_new", num(MAX_NEW)),
                ]),
            ),
        ]),
    )
}

fn new_model(app: &App, req: &Request) -> Response {
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
    match fresh_model(name, seed) {
        Ok(m) => {
            let out = model_json(&m);
            *lock(&app.model) = Some(m);
            Response::json(200, &out)
        }
        Err(e) => Response::error(400, &e),
    }
}

fn train(app: &Arc<App>, req: &Request) -> Response {
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
    match parsed.and_then(|s| {
        app.start_training(s)
            .map_err(|e| Response::error(if e.contains("already") { 409 } else { 400 }, &e))
    }) {
        Ok(()) => Response::json(200, &Json::obj([("state", Json::Str("running".into()))])),
        Err(r) => r,
    }
}

fn train_status(app: &App) -> Response {
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
        ]),
    )
}

fn generate(app: &App, req: &Request) -> Response {
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
        Some(p) if p.is_empty() => return Response::error(400, "the prompt is empty"),
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
    match app
        .backend
        .generate(&model, &tokenizer::encode(prompt), &opts)
    {
        Ok(ids) => Response::json(
            200,
            &Json::obj([
                (
                    "text",
                    Json::Str(tokenizer::decode(&ids).unwrap_or_default()),
                ),
                ("tokens", Json::Arr(ids.iter().map(|&t| num(t)).collect())),
                ("device", Json::Str(app.backend.kind().into())),
            ]),
        ),
        Err(e) => Response::error(400, &e),
    }
}

fn export(app: &App, req: &Request) -> Response {
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

fn import(app: &App, req: &Request) -> Response {
    if lock(&app.job).state == JobState::Running {
        return Response::error(409, "stop the training run first");
    }
    let parsed = Gguf::parse(req.body.clone())
        .map_err(|e| e.to_string())
        .and_then(|g| from_gguf(&g).map_err(|e| e.to_string()));
    match parsed {
        Ok(m) if m.num_params() > MAX_PARAMS => Response::error(
            413,
            &format!(
                "the model has {} parameters; the GUI holds at most {MAX_PARAMS}",
                m.num_params()
            ),
        ),
        Ok(m) => {
            let out = model_json(&m);
            *lock(&app.model) = Some(m);
            Response::json(200, &out)
        }
        Err(e) => Response::error(400, &format!("not a usable llama/qwen2 GGUF: {e}")),
    }
}

/// Route one request.
pub fn route(app: &Arc<App>, req: &Request) -> Response {
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
        ("GET", "/api/selftest/result") => match lock(&app.selftest).clone() {
            Some(s) => Response::new(200, "application/json", s),
            None => Response::error(404, "no self-test result yet"),
        },
        ("GET", _) => assets::serve(p).unwrap_or_else(|| Response::error(404, "not found")),
        (_, p) if p.starts_with("/api/") => Response::error(405, "method not allowed"),
        _ => Response::error(405, "method not allowed"),
    }
}
