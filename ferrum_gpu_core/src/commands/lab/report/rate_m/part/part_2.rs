use super::*;

impl Report {
    /// The report as JSON (the file to send back).
    pub fn json(&self) -> Json {
        let s = |v: &str| Json::Str(v.to_string());
        let adapter = self.adapter.as_ref().map_or(Json::Null, |a| {
            Json::obj([
                ("name", s(&a.name)),
                ("backend", s(&a.backend)),
                ("kind", s(&a.kind)),
                ("driver", s(&a.driver)),
                ("max_binding_mib", Json::Num(a.max_binding_mib as f64)),
                (
                    "features",
                    Json::Arr(a.features.iter().map(|f| s(f)).collect()),
                ),
                ("budget", s(&a.budget)),
            ])
        });
        Json::obj([
            ("format", Json::Num(1.0)),
            ("when", s(&self.when)),
            ("os", s(&self.os)),
            ("wgpu", s(&self.wgpu)),
            ("adapter", adapter),
            ("model", s(&self.model)),
            ("reps", Json::Num(self.reps as f64)),
            (
                "checks",
                Json::Arr(
                    self.checks
                        .iter()
                        .map(|c| {
                            Json::obj([
                                ("name", s(&c.name)),
                                ("error", Json::Num(f64::from(c.error))),
                                ("tolerance", Json::Num(f64::from(c.tolerance))),
                                ("ok", Json::Bool(c.ok())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "speeds",
                Json::Arr(
                    self.speeds
                        .iter()
                        .map(|p| {
                            Json::obj([
                                ("precision", s(&p.precision)),
                                ("model_mib", Json::Num(p.model_mib)),
                                ("decode_tps", Json::Num(p.decode_tps)),
                                ("prefill_tps", Json::Num(p.prefill_tps)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "cpu_prefill_tps",
                self.cpu_prefill_tps.map_or(Json::Null, Json::Num),
            ),
            (
                "notes",
                Json::Arr(self.notes.iter().map(|n| s(n)).collect()),
            ),
            ("passed", Json::Bool(self.passed())),
        ])
    }
}
