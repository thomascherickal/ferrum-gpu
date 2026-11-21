use super::*;

impl Report {
    /// True if every parity check passed (and there was at least one).
    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(Check::ok)
    }

    /// The report as lines of text.
    pub fn text(&self) -> String {
        let mut l = vec![format!(
            "lab report {} on {} (wgpu {})",
            self.when, self.os, self.wgpu
        )];
        match &self.adapter {
            Some(a) => {
                l.push(format!("adapter: {} ({}, {})", a.name, a.kind, a.backend));
                l.push(format!(
                    "  driver: {}",
                    if a.driver.is_empty() {
                        "not reported"
                    } else {
                        &a.driver
                    }
                ));
                l.push(format!(
                    "  features: {}; largest buffer {} MiB",
                    if a.features.is_empty() {
                        "none reported".to_string()
                    } else {
                        a.features.join(", ")
                    },
                    a.max_binding_mib
                ));
                l.push(format!("  {}", a.budget));
            }
            None => l.push("adapter: none (no GPU was used)".into()),
        }
        l.push("parity against the CPU reference:".into());
        for c in &self.checks {
            l.push(format!(
                "  {} {}: max relative error {:.2e} (allowed {:.0e})",
                if c.ok() { "PASS" } else { "FAIL" },
                c.name,
                c.error,
                c.tolerance
            ));
        }
        if !self.speeds.is_empty() {
            l.push(format!(
                "speed ({}; median of {} runs):",
                self.model, self.reps
            ));
            for s in &self.speeds {
                l.push(format!(
                    "  {}: model buffers {:.1} MiB, decode {} tokens/s, prefill {} tokens/s",
                    s.precision,
                    s.model_mib,
                    rate(s.decode_tps),
                    rate(s.prefill_tps)
                ));
            }
            if let Some(c) = self.cpu_prefill_tps {
                l.push(format!(
                    "  CPU reference prefill, f32: {} tokens/s",
                    rate(c)
                ));
            }
        }
        for n in &self.notes {
            l.push(format!("note: {n}"));
        }
        l.push(format!(
            "result: {}",
            if self.passed() { "PASS" } else { "FAIL" }
        ));
        l.join("\n")
    }

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

/// Tokens per second written without false precision: `1234`, `56`, `7.8`.
pub(super) fn rate(v: f64) -> String {
    if !v.is_finite() || v < 0.0 {
        "n/a".into()
    } else if v >= 10.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}
