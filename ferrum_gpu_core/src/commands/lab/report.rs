//! The lab report: what a machine's adapter is, whether the GPU agrees with the CPU reference, and
//! how fast it is. Plain data with a text and a JSON rendering; nothing here touches a device, so it
//! is fully tested, and an owner of hardware we do not have can run `ferrum-gpu lab --json` and
//! send the file back.

use crate::json::Json;

/// The adapter the report was measured on.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterFacts {
    /// Name as the adapter or the check reports it.
    pub name: String,
    /// Graphics API in use (Vulkan, Metal, Dx12, ...).
    pub backend: String,
    /// Discrete, Integrated, Software or Other.
    pub kind: String,
    /// Driver name and version as wgpu reports them (empty if not reported).
    pub driver: String,
    /// Largest storage buffer binding, in MiB.
    pub max_binding_mib: u64,
    /// Optional capabilities the adapter reports (f16, subgroups, ...).
    pub features: Vec<String>,
    /// The memory budget sentence of `info`.
    pub budget: String,
}

/// One parity check: the GPU's answer against the CPU reference.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// Name as the adapter or the check reports it.
    pub name: String,
    /// Largest relative error seen.
    pub error: f32,
    /// The error the check allows.
    pub tolerance: f32,
}

impl Check {
    /// Whether the error is within the tolerance (a NaN error is never within it).
    pub fn ok(&self) -> bool {
        self.error < self.tolerance
    }
}

/// Speed at one precision.
#[derive(Clone, Debug, PartialEq)]
pub struct Speed {
    /// The weight precision (f32, Q8_0, Q4_0).
    pub precision: String,
    /// Device memory the model's buffers plan to occupy.
    pub model_mib: f64,
    /// Tokens per second generating one token at a time.
    pub decode_tps: f64,
    /// Tokens per second processing a whole sequence at once.
    pub prefill_tps: f64,
}

/// A whole report.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Report {
    /// UTC date, `YYYY-MM-DD`.
    pub when: String,
    /// Operating system and architecture.
    pub os: String,
    /// The wgpu version this build links.
    pub wgpu: String,
    /// The adapter measured, if a GPU was used.
    pub adapter: Option<AdapterFacts>,
    /// A sentence about the model the speeds were measured on.
    pub model: String,
    /// Timed repetitions behind each median.
    pub reps: usize,
    /// The parity checks, in the order run.
    pub checks: Vec<Check>,
    /// Speed per precision.
    pub speeds: Vec<Speed>,
    /// The CPU reference's prefill speed on the same model, for scale.
    pub cpu_prefill_tps: Option<f64>,
    /// Caveats the reader needs (what the numbers are not).
    pub notes: Vec<String>,
}

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
fn rate(v: f64) -> String {
    if !v.is_finite() || v < 0.0 {
        "n/a".into()
    } else if v >= 10.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// `YYYY-MM-DD` (UTC) for `secs` since the Unix epoch (days since 1970 turned into a calendar date).
pub fn date_of(secs: u64) -> String {
    let z = (secs / 86400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The version of crate `name` in a `Cargo.lock` text, or `unknown`.
pub fn locked_version(lock: &str, name: &str) -> String {
    let head = format!("name = \"{name}\"");
    let mut lines = lock.lines();
    while let Some(l) = lines.next() {
        if l.trim() == head {
            if let Some(v) = lines
                .next()
                .and_then(|v| v.trim().strip_prefix("version = \""))
            {
                return v.trim_end_matches('"').to_string();
            }
        }
    }
    "unknown".into()
}

#[cfg(test)]
mod tests;
