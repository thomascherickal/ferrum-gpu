//! The `--device auto` rule, from the measured crossovers (`docs/train-speed.md`: batched
//! training is 1.3x to 5.5x faster on the GPU; `docs/decode-crossover.md`: single-token decode is
//! not faster on the GPU up to 34M parameters). A pure function over an adapter table, so tests
//! use fake adapters.

use ferrum_gpu_gui::app::DeviceChoice;
use ferrum_wgpu::device::select::{choose, AdapterSummary};

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
    /// What to ask `App::new` for.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_wgpu::device::select::Kind;

    fn ad(name: &str, kind: Kind) -> AdapterSummary {
        AdapterSummary {
            name: name.into(),
            backend: "Vulkan".into(),
            kind,
            max_buffer: 1 << 30,
            max_binding: 1 << 27,
            max_workgroup_invocations: 256,
        }
    }

    fn table() -> Vec<AdapterSummary> {
        vec![
            ad("llvmpipe", Kind::Software),
            ad("Xe", Kind::Integrated),
            ad("MX330", Kind::Discrete),
        ]
    }

    #[test]
    fn cpu_is_always_cpu() {
        for w in [Workload::Train, Workload::Decode] {
            let d = decide(DeviceChoice::Cpu, w, &table(), None).unwrap();
            assert_eq!(
                (d.choice, d.adapter, d.note),
                (DeviceChoice::Cpu, None, None)
            );
        }
    }

    #[test]
    fn explicit_gpu_picks_the_best_or_the_named_adapter_or_fails() {
        let d = decide(DeviceChoice::Gpu, Workload::Decode, &table(), None).unwrap();
        assert_eq!(
            (d.choice, d.adapter.as_deref()),
            (DeviceChoice::Gpu, Some("MX330"))
        );
        let d = decide(DeviceChoice::Gpu, Workload::Train, &table(), Some("llvm")).unwrap();
        assert_eq!(d.adapter.as_deref(), Some("llvmpipe"));
        assert!(decide(DeviceChoice::Gpu, Workload::Train, &[], None)
            .unwrap_err()
            .contains("no GPU"));
        assert!(
            decide(DeviceChoice::Gpu, Workload::Train, &table(), Some("zzz"))
                .unwrap_err()
                .contains("no adapter matches")
        );
    }

    #[test]
    fn auto_trains_on_hardware_and_decodes_on_the_cpu() {
        let d = decide(DeviceChoice::Auto, Workload::Train, &table(), None).unwrap();
        assert_eq!(
            (d.choice, d.adapter.as_deref(), d.note),
            (DeviceChoice::Auto, Some("MX330"), None)
        );
        let d = decide(DeviceChoice::Auto, Workload::Decode, &table(), None).unwrap();
        assert_eq!(d.choice, DeviceChoice::Cpu);
        assert!(d.note.unwrap().contains("34M"));
    }

    #[test]
    fn auto_falls_back_to_the_cpu_with_one_line_when_there_is_no_hardware() {
        for (list, why) in [
            (vec![], "no GPU adapters"),
            (vec![ad("llvmpipe", Kind::Software)], "only software"),
        ] {
            let d = decide(DeviceChoice::Auto, Workload::Train, &list, None).unwrap();
            assert_eq!((d.choice, d.adapter), (DeviceChoice::Cpu, None));
            let n = d.note.unwrap();
            assert!(
                n.contains(why) && n.ends_with("using the CPU") && !n.contains('\n'),
                "{n}"
            );
        }
    }
}
