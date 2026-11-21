use super::*;

pub(super) fn system(app: &App) -> Response {
    // the machine's RAM does not depend on the adapter; a stand-in adapter reads it through the same readers
    let ram = facts_for(&cpu_like(), &System);
    let adapters = app
        .adapters
        .iter()
        .map(|a| {
            let (budget, facts) = budget_for(None, a, &System);
            let fits = budget.map(|b| {
                let bytes = b.bytes as f64;
                Json::obj([
                    (
                        "infer",
                        Json::Arr(
                            Precision::ALL
                                .iter()
                                .map(|p| {
                                    Json::obj([
                                        ("precision", Json::Str(p.name().into())),
                                        (
                                            "text",
                                            Json::Str(format_params(max_params_inference(
                                                bytes, *p,
                                            ))),
                                        ),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                    (
                        "train",
                        Json::Str(format_params(max_params_training(bytes))),
                    ),
                ])
            });
            Json::obj([
                ("name", Json::Str(a.name.clone())),
                ("kind", Json::Str(format!("{:?}", a.kind))),
                ("backend", Json::Str(a.backend.clone())),
                (
                    "budget_mib",
                    budget.map_or(Json::Null, |b| Json::Num((b.bytes >> 20) as f64)),
                ),
                (
                    "note",
                    Json::Str(ferrum_gpu_core::commands::preflight::explain(
                        budget, &facts,
                    )),
                ),
                ("fits", fits.unwrap_or(Json::Null)),
                ("meter", meter::meter_json(&facts, a.kind)),
            ])
        })
        .collect();
    let num = |v: Option<u64>| v.map_or(Json::Null, |b| Json::Num(b as f64));
    Response::json(
        200,
        &Json::obj([
            ("os", Json::Str(std::env::consts::OS.into())),
            ("arch", Json::Str(std::env::consts::ARCH.into())),
            (
                "cpu_threads",
                Json::Num(std::thread::available_parallelism().map_or(1, |n| n.get()) as f64),
            ),
            ("ram_total", num(ram.ram_total)),
            ("ram_available", num(ram.ram_available)),
            ("adapters", Json::Arr(adapters)),
        ]),
    )
}

/// A stand-in adapter used only to read the machine's RAM through the same readers.
pub(super) fn cpu_like() -> ferrum_wgpu::select::AdapterSummary {
    ferrum_wgpu::select::AdapterSummary {
        name: "ram".into(),
        backend: "none".into(),
        kind: ferrum_wgpu::select::Kind::Other,
        max_buffer: 0,
        max_binding: 0,
        max_workgroup_invocations: 0,
        features: Default::default(),
    }
}
