//! Pure adapter selection over a list of summaries (testable without hardware).

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

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectError::NoAdapters => write!(f, "no GPU adapters found"),
            SelectError::NoMatch { wanted, available } => {
                write!(
                    f,
                    "no adapter matches '{wanted}'; available: {}",
                    available.join(", ")
                )
            }
            SelectError::OnlySoftware => {
                write!(
                    f,
                    "only software adapters found; name one explicitly with --gpu-adapter"
                )
            }
        }
    }
}

impl std::error::Error for SelectError {}

/// Choose an adapter index. A name fragment (case-insensitive) wins and may pick a software
/// adapter. Otherwise: first discrete, else first integrated, else first other; software is
/// never chosen unprompted.
pub fn choose(list: &[AdapterSummary], name: Option<&str>) -> Result<usize, SelectError> {
    if list.is_empty() {
        return Err(SelectError::NoAdapters);
    }
    if let Some(want) = name {
        let w = want.to_lowercase();
        return list
            .iter()
            .position(|a| a.name.to_lowercase().contains(&w))
            .ok_or_else(|| SelectError::NoMatch {
                wanted: want.to_string(),
                available: list.iter().map(|a| a.name.clone()).collect(),
            });
    }
    for kind in [Kind::Discrete, Kind::Integrated, Kind::Other] {
        if let Some(i) = list.iter().position(|a| a.kind == kind) {
            return Ok(i);
        }
    }
    Err(SelectError::OnlySoftware)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(name: &str, kind: Kind) -> AdapterSummary {
        AdapterSummary {
            name: name.into(),
            backend: "Vulkan".into(),
            kind,
            max_buffer: 1 << 30,
            max_binding: 1 << 27,
            max_workgroup_invocations: 256,
        }
    }

    #[test]
    fn prefers_discrete_over_integrated() {
        let l = [
            a("Intel UHD", Kind::Integrated),
            a("NVIDIA MX330", Kind::Discrete),
        ];
        assert_eq!(choose(&l, None), Ok(1));
    }

    #[test]
    fn falls_back_to_integrated_then_other() {
        assert_eq!(
            choose(
                &[
                    a("llvmpipe", Kind::Software),
                    a("Apple M2", Kind::Integrated)
                ],
                None
            ),
            Ok(1)
        );
        assert_eq!(choose(&[a("x", Kind::Other)], None), Ok(0));
    }

    #[test]
    fn never_picks_software_unprompted() {
        assert_eq!(
            choose(&[a("llvmpipe", Kind::Software)], None),
            Err(SelectError::OnlySoftware)
        );
    }

    #[test]
    fn a_name_picks_software_and_is_case_insensitive() {
        let l = [
            a("NVIDIA MX330", Kind::Discrete),
            a("llvmpipe (LLVM 17)", Kind::Software),
        ];
        assert_eq!(choose(&l, Some("LLVMPIPE")), Ok(1));
    }

    #[test]
    fn unknown_name_lists_what_exists() {
        let l = [a("NVIDIA MX330", Kind::Discrete)];
        let e = choose(&l, Some("radeon")).unwrap_err();
        assert!(e.to_string().contains("NVIDIA MX330") && e.to_string().contains("radeon"));
    }

    #[test]
    fn empty_list_is_an_error() {
        let e = choose(&[], None).unwrap_err();
        assert_eq!(e, SelectError::NoAdapters);
        assert!(e.to_string().contains("no GPU"));
        assert!(choose(&[], Some("x")).is_err());
        assert!(SelectError::OnlySoftware.to_string().contains("software"));
    }
}
