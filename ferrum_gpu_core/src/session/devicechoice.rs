use super::*;

/// Which device the user asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The CPU oracle.
    Cpu,
    /// A GPU adapter; an error if none can be opened.
    Gpu,
    /// A GPU if one opens, else the CPU.
    Auto,
}

impl DeviceChoice {
    /// Parse `cpu`, `gpu` or `auto`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(format!("unknown device '{other}' (cpu, gpu, auto)")),
        }
    }

    /// The flag spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Auto => "auto",
        }
    }
}

/// An opened device: the backend, what was asked for, and the adapters the machine reports.
pub struct Session {
    /// Where models run.
    pub backend: Arc<dyn Backend>,
    /// What the user asked for.
    pub requested: DeviceChoice,
    /// Why the backend is not the requested one, if it is not.
    pub fallback_note: Option<String>,
    /// Every adapter the machine reports.
    pub adapters: Vec<AdapterSummary>,
    /// How a GPU decoder stores its weights (`None` is f32); the pre-flight plans for it.
    pub weights: Option<QKind>,
}
