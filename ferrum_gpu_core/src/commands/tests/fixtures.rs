use super::*;
use crate::engine::{Backend, GenOpts, Progress, TrainSpec};
use ferrum_wgpu::budget::BufferPlan;
use ferrum_wgpu::select::{AdapterSummary, Kind};

/// Pretends to be a GPU named "Fake GPU"; never runs anything.
pub(super) struct FakeGpu;
impl Backend for FakeGpu {
    fn kind(&self) -> &'static str {
        "gpu"
    }
    fn name(&self) -> String {
        "Fake GPU".into()
    }
    fn train(&self, _: &Llama, _: &TrainSpec, _: &Arc<Mutex<Progress>>) -> Result<Llama, String> {
        unreachable!()
    }
    fn generate(&self, _: &Llama, _: &[usize], _: &GenOpts) -> Result<Vec<usize>, String> {
        unreachable!()
    }
}

pub(super) fn listed() -> Vec<AdapterSummary> {
    vec![AdapterSummary {
        name: "Fake GPU".into(),
        backend: "Vulkan".into(),
        kind: Kind::Discrete,
        max_buffer: 1 << 32,
        max_binding: 1 << 30,
        max_workgroup_invocations: 256,
        features: Default::default(),
    }]
}

pub(super) fn session_on(b: Arc<dyn Backend>, adapters: Vec<AdapterSummary>) -> Session {
    Session::with_backend(b, DeviceChoice::Gpu, None, adapters)
}

/// A card with 2048 MiB total and 1989 MiB free, as `nvidia-smi` would say; no RAM figures.
pub(super) struct Mx330;
impl sysmem::Readers for Mx330 {
    fn nvidia_smi(&self) -> Option<String> {
        Some("Fake GPU, 2048, 1989\n".into())
    }
    fn ram_bytes(&self) -> (Option<u64>, Option<u64>) {
        (None, None)
    }
    fn sysfs_vram_mb(&self) -> Vec<u64> {
        vec![]
    }
}

/// A plan of `mib` MiB in two buffers (each under the fake adapter's binding limit).
pub(super) fn plan_of(mib: u64) -> BufferPlan {
    let mut p = BufferPlan::default();
    p.add("first half", mib / 2 * 1048576);
    p.add("second half", (mib - mib / 2) * 1048576);
    p
}

pub(super) struct Blind;
impl sysmem::Readers for Blind {
    fn nvidia_smi(&self) -> Option<String> {
        None
    }
    fn ram_bytes(&self) -> (Option<u64>, Option<u64>) {
        (None, None)
    }
    fn sysfs_vram_mb(&self) -> Vec<u64> {
        vec![]
    }
}
