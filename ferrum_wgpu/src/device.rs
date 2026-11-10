//! Real adapter enumeration and device opening. All direct wgpu device calls live in this file,
//! `tensor.rs` and `matmul.rs` so a wgpu upgrade touches three places.

use crate::select::{choose, AdapterSummary, Kind, SelectError};
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

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

fn kind_of(t: wgpu::DeviceType) -> Kind {
    match t {
        wgpu::DeviceType::DiscreteGpu => Kind::Discrete,
        wgpu::DeviceType::IntegratedGpu => Kind::Integrated,
        wgpu::DeviceType::Cpu => Kind::Software,
        _ => Kind::Other,
    }
}

fn summarize(a: &wgpu::Adapter) -> AdapterSummary {
    let i = a.get_info();
    let l = a.limits();
    AdapterSummary {
        name: i.name,
        backend: format!("{:?}", i.backend),
        kind: kind_of(i.device_type),
        max_buffer: l.max_buffer_size,
        max_binding: l.max_storage_buffer_binding_size,
        max_workgroup_invocations: l.max_compute_invocations_per_workgroup,
    }
}

/// An opened device, its queue, and the adapter facts.
pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    info: AdapterSummary,
    max_workgroups: u32,
    /// Set by the device-lost / uncaptured-error callbacks; see `Gpu::check`.
    pub(crate) health: Arc<Mutex<Option<String>>>,
    pub(crate) pipelines: Mutex<HashMap<&'static str, wgpu::ComputePipeline>>,
    pub(crate) pipelines_built: AtomicUsize,
    pub(crate) buffers_created: AtomicUsize,
    /// The open command batch, if any (see `Gpu::begin_batch`).
    pub(crate) batch: Mutex<Option<wgpu::CommandEncoder>>,
}

impl Gpu {
    /// Every adapter wgpu can see (empty on a machine without drivers).
    pub fn list() -> Vec<AdapterSummary> {
        let inst = wgpu::Instance::default();
        block_on(inst.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .map(summarize)
            .collect()
    }

    /// Open an adapter chosen by [`choose`] (`name` is a case-insensitive fragment). The device
    /// is requested with the adapter's own limits, so nothing is assumed beyond what it reports.
    pub fn open(name: Option<&str>) -> Result<Gpu, GpuError> {
        let inst = wgpu::Instance::default();
        let mut adapters = block_on(inst.enumerate_adapters(wgpu::Backends::all()));
        let summaries: Vec<_> = adapters.iter().map(summarize).collect();
        let idx = choose(&summaries, name)?;
        let adapter = adapters.swap_remove(idx);
        let limits = adapter.limits();
        let desc = wgpu::DeviceDescriptor {
            label: Some("ferrum"),
            required_limits: limits.clone(),
            ..Default::default()
        };
        let (device, queue) = block_on(adapter.request_device(&desc))
            .map_err(|e| GpuError(format!("request_device failed: {e}")))?;
        // Device loss and uncaptured errors are recorded, never allowed to abort the process.
        let health = Arc::new(Mutex::new(None));
        let h = Arc::clone(&health);
        device.set_device_lost_callback(move |reason, msg| {
            *h.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(format!("GPU device lost: {reason:?}: {msg}"));
        });
        let h = Arc::clone(&health);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            *h.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("GPU error: {e}"));
        }));
        Ok(Gpu {
            device,
            queue,
            info: summaries[idx].clone(),
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            health,
            pipelines: Mutex::new(HashMap::new()),
            pipelines_built: AtomicUsize::new(0),
            buffers_created: AtomicUsize::new(0),
            batch: Mutex::new(None),
        })
    }

    /// Facts about the opened adapter.
    pub fn info(&self) -> &AdapterSummary {
        &self.info
    }

    /// Largest workgroup count per dispatch dimension.
    pub fn max_workgroups(&self) -> u32 {
        self.max_workgroups
    }
}

/// Open a GPU for a test, or print why not and return `None`. `FERRUM_GPU_ADAPTER` names an
/// adapter (e.g. `llvmpipe`); otherwise a hardware adapter is used when one exists.
pub fn open_for_test() -> Option<Gpu> {
    let want = std::env::var("FERRUM_GPU_ADAPTER").ok();
    match Gpu::open(want.as_deref()) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter ({e})");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_on_runs_ready_and_pending_futures() {
        assert_eq!(block_on(async { 7 }), 7);
        struct Twice(u8);
        impl Future for Twice {
            type Output = u8;
            fn poll(mut self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<u8> {
                self.0 += 1;
                if self.0 >= 3 {
                    Poll::Ready(self.0)
                } else {
                    Poll::Pending
                }
            }
        }
        assert_eq!(block_on(Twice(0)), 3);
    }

    #[test]
    fn kind_mapping() {
        assert_eq!(kind_of(wgpu::DeviceType::DiscreteGpu), Kind::Discrete);
        assert_eq!(kind_of(wgpu::DeviceType::IntegratedGpu), Kind::Integrated);
        assert_eq!(kind_of(wgpu::DeviceType::Cpu), Kind::Software);
        assert_eq!(kind_of(wgpu::DeviceType::VirtualGpu), Kind::Other);
    }

    #[test]
    fn listing_never_panics_and_unknown_name_errors() {
        let list = Gpu::list();
        eprintln!("adapters: {list:?}");
        let e = Gpu::open(Some("no-such-adapter-xyz"))
            .err()
            .expect("must fail");
        assert!(e.to_string().contains("no-such-adapter-xyz") || e.to_string().contains("no GPU"));
    }

    #[test]
    fn open_reports_limits_when_a_gpu_exists() {
        let Some(g) = open_for_test() else { return };
        assert!(g.info().max_binding > 0);
        assert!(g.max_workgroups() >= 65535);
        assert!(g.info().max_workgroup_invocations >= 256);
    }
}
