/// What kind of device an adapter is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Discrete GPU.
    Discrete,
    /// Integrated or unified-memory GPU (Intel iGPU, AMD APU, Apple Silicon).
    Integrated,
    /// Software rasteriser (lavapipe, WARP, SwiftShader).
    Software,
    /// Anything else.
    Other,
}

/// Optional capabilities an adapter reports. They are only *reported*: a feature is requested from
/// the device when a kernel that uses it exists, never speculatively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdapterFeatures {
    /// 16-bit floats in shaders (`SHADER_F16`): half the memory and bandwidth for weights and caches.
    pub shader_f16: bool,
    /// Subgroup operations (`SUBGROUP`): the base of fast reductions and matrix kernels.
    pub subgroups: bool,
    /// Cooperative (matrix-multiply-accumulate) operations, still experimental in `wgpu`.
    pub cooperative_matrix: bool,
    /// GPU timestamp queries, for timing kernels on the device.
    pub timestamp_query: bool,
}

impl AdapterFeatures {
    /// The names of the features present, for display (`f16`, `subgroups`, `cooperative-matrix`, `timestamps`).
    pub fn words(&self) -> Vec<&'static str> {
        [
            (self.shader_f16, "f16"),
            (self.subgroups, "subgroups"),
            (self.cooperative_matrix, "cooperative-matrix"),
            (self.timestamp_query, "timestamps"),
        ]
        .into_iter()
        .filter_map(|(on, w)| on.then_some(w))
        .collect()
    }
}

/// The facts selection and budgeting need about one adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    /// Adapter name as the driver reports it.
    pub name: String,
    /// Backend name (`Vulkan`, `Metal`, `Dx12`, ...).
    pub backend: String,
    /// Device kind.
    pub kind: Kind,
    /// Granted `max_buffer_size`.
    pub max_buffer: u64,
    /// Granted `max_storage_buffer_binding_size`.
    pub max_binding: u64,
    /// Granted `max_compute_invocations_per_workgroup`.
    pub max_workgroup_invocations: u32,
    /// Optional capabilities the adapter reports.
    pub features: AdapterFeatures,
}

/// Why no adapter was chosen.
#[derive(Debug, PartialEq, Eq)]
pub enum SelectError {
    /// No adapters exist at all.
    NoAdapters,
    /// The named adapter matched nothing; `available` lists what exists.
    NoMatch {
        /// The requested name fragment.
        wanted: String,
        /// Names that do exist.
        available: Vec<String>,
    },
    /// Adapters exist but only software ones, and none was named.
    OnlySoftware,
}
