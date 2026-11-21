use super::*;

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
                    "only software or OpenGL adapters found; name one explicitly with --gpu-adapter"
                )
            }
        }
    }
}

impl std::error::Error for SelectError {}

/// Choose an adapter index. A name fragment (case-insensitive) wins and may pick a software
/// adapter. Otherwise: first discrete, else first integrated, else first other; software and
/// OpenGL-backend adapters are never chosen unprompted (Phase 7: the Mesa GL adapter failed the GELU
/// parity test, see docs/devices.md).
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
        if let Some(i) = list
            .iter()
            .position(|a| a.kind == kind && a.backend != "Gl")
        {
            return Ok(i);
        }
    }
    Err(SelectError::OnlySoftware)
}
