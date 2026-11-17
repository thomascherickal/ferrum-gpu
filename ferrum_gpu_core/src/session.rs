//! Choosing a device and holding the backend that was opened: what every front end needs before it
//! can train or generate.

use crate::engine::{Backend, CpuBackend, GpuBackend};
use ferrum_wgpu::device::select::AdapterSummary;
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::Gpu;
use std::sync::Arc;

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

impl Session {
    /// Open the device for `choice`. `adapter` names a GPU (fragment). `Auto` falls back to the CPU
    /// with a note; `Gpu` with nothing usable is an error.
    pub fn new(
        choice: DeviceChoice,
        adapter: Option<&str>,
        cpu_threads: usize,
    ) -> Result<Session, String> {
        Session::new_with(choice, adapter, cpu_threads, None)
    }

    /// [`new`](Session::new) with the weight storage of a GPU decoder chosen.
    pub fn new_with(
        choice: DeviceChoice,
        adapter: Option<&str>,
        cpu_threads: usize,
        weights: Option<QKind>,
    ) -> Result<Session, String> {
        let adapters = Gpu::list();
        let mut note = None;
        let backend: Arc<dyn Backend> = match choice {
            DeviceChoice::Cpu => Arc::new(CpuBackend::new(cpu_threads)),
            DeviceChoice::Gpu => Arc::new(GpuBackend::open(adapter)?.with_weights(weights)),
            DeviceChoice::Auto => match GpuBackend::open(adapter) {
                Ok(g) => Arc::new(g.with_weights(weights)),
                Err(e) => {
                    note = Some(format!("no usable GPU ({e}); using the CPU"));
                    Arc::new(CpuBackend::new(cpu_threads))
                }
            },
        };
        let mut s = Session::with_backend(backend, choice, note, adapters);
        s.weights = weights;
        Ok(s)
    }

    /// A session around an already-built backend (tests use this with fakes).
    pub fn with_backend(
        backend: Arc<dyn Backend>,
        requested: DeviceChoice,
        fallback_note: Option<String>,
        adapters: Vec<AdapterSummary>,
    ) -> Session {
        Session {
            backend,
            requested,
            fallback_note,
            adapters,
            weights: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_choices_parse_and_print_the_same_words() {
        for (s, c) in [
            ("cpu", DeviceChoice::Cpu),
            ("gpu", DeviceChoice::Gpu),
            ("auto", DeviceChoice::Auto),
        ] {
            assert_eq!(DeviceChoice::parse(s), Ok(c));
            assert_eq!(c.as_str(), s);
        }
        let e = DeviceChoice::parse("tpu").unwrap_err();
        assert!(e.contains("'tpu'") && e.contains("cpu, gpu, auto"), "{e}");
        assert!(DeviceChoice::parse("").is_err());
        assert!(DeviceChoice::parse("CPU").is_err(), "the spelling is exact");
    }

    #[test]
    fn a_cpu_session_has_a_cpu_backend_and_no_note() {
        let s = Session::new(DeviceChoice::Cpu, None, 2).unwrap();
        assert_eq!(s.backend.kind(), "cpu");
        assert_eq!(s.requested, DeviceChoice::Cpu);
        assert_eq!(s.fallback_note, None);
    }

    #[test]
    fn auto_falls_back_to_the_cpu_with_a_note_when_the_named_adapter_is_absent() {
        let s = Session::new(DeviceChoice::Auto, Some("no-such-adapter-zzz"), 1).unwrap();
        assert_eq!(s.backend.kind(), "cpu");
        let note = s.fallback_note.expect("a note");
        assert!(
            note.contains("no usable GPU") && note.contains("using the CPU"),
            "{note}"
        );
        assert_eq!(s.requested, DeviceChoice::Auto);
    }

    #[test]
    fn gpu_with_a_missing_adapter_is_an_error_not_a_silent_cpu() {
        let e = Session::new(DeviceChoice::Gpu, Some("no-such-adapter-zzz"), 1)
            .err()
            .expect("an error");
        assert!(!e.is_empty());
    }

    #[test]
    fn with_backend_keeps_what_it_is_given() {
        let b: Arc<dyn Backend> = Arc::new(CpuBackend::new(1));
        let s = Session::with_backend(b, DeviceChoice::Gpu, Some("note".into()), vec![]);
        assert_eq!(s.requested, DeviceChoice::Gpu);
        assert_eq!(s.fallback_note.as_deref(), Some("note"));
        assert!(s.adapters.is_empty());
    }
}
