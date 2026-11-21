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
}
