use super::*;

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

pub(super) fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

pub(super) fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))
}

/// An optional number field: absent gives `default`, wrong type or range is an error.
pub(super) fn field(j: &Json, key: &str, default: f64, lo: f64, hi: f64) -> Result<f64, Response> {
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

pub(super) fn model_json(m: &Llama) -> Json {
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

pub(super) fn info(app: &App) -> Response {
    let adapters = app
        .adapters
        .iter()
        .map(|a| {
            let (budget, facts) = sysmem::budget_for(None, a, &sysmem::System);
            Json::obj([
                ("name", Json::Str(a.name.clone())),
                ("backend", Json::Str(a.backend.clone())),
                ("kind", Json::Str(format!("{:?}", a.kind))),
                ("max_binding_mib", num((a.max_binding / (1 << 20)) as usize)),
                (
                    "features",
                    Json::Arr(
                        a.features
                            .words()
                            .into_iter()
                            .map(|w| Json::Str(w.into()))
                            .collect(),
                    ),
                ),
                (
                    "memory_budget_mib",
                    budget.map_or(Json::Null, |b| num((b.bytes >> 20) as usize)),
                ),
                ("memory_note", Json::Str(preflight::explain(budget, &facts))),
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
    let version = app.version.load(std::sync::atomic::Ordering::SeqCst);
    let model = lock(&app.model).as_ref().map_or(Json::Null, |m| {
        let mut j = model_json(m);
        if let Json::Obj(o) = &mut j {
            o.push(("version".into(), num(version as usize)));
        }
        j
    });
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
