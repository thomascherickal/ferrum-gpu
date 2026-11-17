//! The Capable tab's API: what a given amount of memory can hold, and what this machine can hold.
//! The formulas are `ferrum_gpu_core::capacity` (the ones in `HARDWARE-CAPs.md`); everything is an estimate.
//!
//! | route | answers |
//! |---|---|
//! | `GET /api/capable?memory_gb=16&unified=0&bandwidth=448&model_b=7` | the largest model per precision for that memory, and the decode-speed ceiling for a model of `model_b` billion weights at `bandwidth` GB/s (both optional) |
//! | `GET /api/system` | this machine: OS, CPU threads, RAM, and for each adapter its memory budget and what that budget can hold |

mod meter;

use crate::app::App;
use crate::http::{Request, Response};
use crate::json::Json;
use ferrum_gpu_core::capacity::{
    decode_ceiling_tps, format_params, max_params_inference, max_params_training, row, row_json,
    Precision,
};
use ferrum_gpu_core::commands::sysmem::{budget_for, facts_for, System};

/// True if `path` belongs to this API.
pub fn handles(path: &str) -> bool {
    path == "/api/capable" || path == "/api/system"
}

fn number(req: &Request, name: &str) -> Result<Option<f64>, Response> {
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

fn capable(req: &Request) -> Response {
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

fn system(app: &App) -> Response {
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
fn cpu_like() -> ferrum_wgpu::select::AdapterSummary {
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

/// Answer a request for this API (see [`handles`]).
pub fn route(app: &App, req: &Request) -> Response {
    if req.method != "GET" {
        return Response::error(405, "method not allowed");
    }
    match req.path.as_str() {
        "/api/capable" => capable(req),
        "/api/system" => system(app),
        _ => Response::error(404, "no such route"),
    }
}

#[cfg(test)]
mod tests;
