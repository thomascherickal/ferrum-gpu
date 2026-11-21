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
