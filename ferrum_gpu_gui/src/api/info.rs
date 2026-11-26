use super::*;

pub(crate) fn info(app: &App) -> Response {
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
        let mut fields = model_fields(m);
        fields.push(("version", num(version as usize)));
        Json::obj_vec(fields)
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
