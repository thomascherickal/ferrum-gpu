use super::*;

/// What the command is about to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    /// Batched forward + backward steps.
    Train,
    /// Single-stream decoding, scoring and everything else (not measured faster on the GPU).
    Decode,
}

/// Where to run, and the one line to print about it, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    /// What to ask `Session::new` for.
    pub choice: DeviceChoice,
    /// The adapter name to open, when the GPU is chosen.
    pub adapter: Option<String>,
    /// A line explaining an `auto` choice that ended on the CPU.
    pub note: Option<String>,
}

/// Decide for `requested` over `adapters`. `name` is the `--gpu-adapter` fragment.
pub fn decide(
    requested: DeviceChoice,
    work: Workload,
    adapters: &[AdapterSummary],
    name: Option<&str>,
) -> Result<Decision, String> {
    let gpu = |i: usize| Decision {
        choice: requested,
        adapter: Some(adapters[i].name.clone()),
        note: None,
    };
    let cpu = |note: Option<String>| Decision {
        choice: DeviceChoice::Cpu,
        adapter: None,
        note,
    };
    match (requested, work) {
        (DeviceChoice::Cpu, _) => Ok(cpu(None)),
        (DeviceChoice::Gpu, _) => choose(adapters, name).map(gpu).map_err(|e| e.to_string()),
        (DeviceChoice::Auto, Workload::Decode) => Ok(cpu(Some(
            "auto: using the CPU for decoding (measured faster than the GPU up to 34M parameters; \
             --device gpu overrides)"
                .into(),
        ))),
        (DeviceChoice::Auto, Workload::Train) => Ok(match choose(adapters, name) {
            Ok(i) => gpu(i),
            Err(e) => cpu(Some(format!("auto: {e}; using the CPU"))),
        }),
    }
}
