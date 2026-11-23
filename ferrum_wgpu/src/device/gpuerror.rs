use super::*;

/// Error from the GPU layer.
#[derive(Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

impl From<SelectError> for GpuError {
    fn from(e: SelectError) -> Self {
        GpuError(e.to_string())
    }
}

/// Run a future to completion on the current thread (no executor dependency). wgpu's native
/// futures complete after a bounded number of polls, so a short sleep between polls is enough.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
}

/// The driver name and version text wgpu reports for each adapter, as `(adapter name, driver)`.
/// Empty text where the driver reports nothing.
pub fn drivers() -> Vec<(String, String)> {
    let inst = wgpu::Instance::default();
    block_on(inst.enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(|a| {
            let i = a.get_info();
            (
                i.name,
                format!("{} {}", i.driver, i.driver_info).trim().to_string(),
            )
        })
        .collect()
}

pub(super) fn kind_of(t: wgpu::DeviceType) -> Kind {
    match t {
        wgpu::DeviceType::DiscreteGpu => Kind::Discrete,
        wgpu::DeviceType::IntegratedGpu => Kind::Integrated,
        wgpu::DeviceType::Cpu => Kind::Software,
        _ => Kind::Other,
    }
}

pub(super) fn summarize(a: &wgpu::Adapter) -> AdapterSummary {
    let i = a.get_info();
    let l = a.limits();
    AdapterSummary {
        name: i.name,
        backend: format!("{:?}", i.backend),
        kind: kind_of(i.device_type),
        max_buffer: l.max_buffer_size,
        max_binding: l.max_storage_buffer_binding_size,
        max_workgroup_invocations: l.max_compute_invocations_per_workgroup,
        features: features_of(a.features()),
    }
}

/// The reported optional capabilities of an adapter, from its `wgpu` feature set.
pub(super) fn features_of(f: wgpu::Features) -> select::AdapterFeatures {
    select::AdapterFeatures {
        shader_f16: f.contains(wgpu::Features::SHADER_F16),
        subgroups: f.contains(wgpu::Features::SUBGROUP),
        cooperative_matrix: f.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
        timestamp_query: f.contains(wgpu::Features::TIMESTAMP_QUERY),
    }
}

/// An opened device, its queue, and the adapter facts.
pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(super) info: AdapterSummary,
    pub(super) max_workgroups: u32,
    /// Set by the device-lost / uncaptured-error callbacks; see `Gpu::check`.
    pub(crate) health: Arc<Mutex<Option<String>>>,
    pub(crate) pipelines: Mutex<HashMap<&'static str, wgpu::ComputePipeline>>,
    pub(crate) pipelines_built: AtomicUsize,
    pub(crate) buffers_created: AtomicUsize,
    /// The open command batch, if any (see `Gpu::begin_batch`).
    pub(crate) batch: Mutex<Option<wgpu::CommandEncoder>>,
    /// Fault injection (tests, or the `fault-injection` feature): how many commands and reads may
    /// still succeed.
    #[cfg(any(test, feature = "fault-injection"))]
    pub(crate) faults: Faults,
}
