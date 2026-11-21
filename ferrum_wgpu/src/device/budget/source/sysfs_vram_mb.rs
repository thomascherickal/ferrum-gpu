use super::*;

/// Total VRAM in MiB of every `<drm>/card<N>/device/mem_info_vram_total` found. The root is
/// injectable so tests use a temp tree; connector entries (`card0-DP-1`) are ignored.
pub fn sysfs_vram_mb(drm: &Path) -> Vec<u64> {
    let mut cards: Vec<_> = std::fs::read_dir(drm)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.strip_prefix("card")
                .is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(|e| e.path())
        .collect();
    cards.sort();
    cards
        .iter()
        .filter_map(|c| std::fs::read_to_string(c.join("device/mem_info_vram_total")).ok())
        .filter_map(|s| parse_vram_mb(&s))
        .collect()
}

/// Resolve the budget: override > tool-reported VRAM > half of RAM when `integrated` > unknown.
pub fn resolve(
    override_mb: Option<u64>,
    tool_mb: Option<u64>,
    integrated: bool,
    ram_bytes: Option<u64>,
) -> Option<Budget> {
    if let Some(mb) = override_mb {
        return Some(Budget {
            bytes: mb * MIB,
            source: Source::Override,
        });
    }
    if let Some(mb) = tool_mb.filter(|_| !integrated) {
        return Some(Budget {
            bytes: mb * MIB,
            source: Source::Tool,
        });
    }
    if integrated {
        return ram_bytes.map(|r| Budget {
            bytes: r / 2,
            source: Source::HalfRam,
        });
    }
    None
}

/// Refusal from [`BufferPlan::check`].
#[derive(Debug, PartialEq, Eq)]
pub enum PlanError {
    /// One buffer is larger than the adapter's binding limit.
    TooLarge {
        /// Buffer name.
        name: String,
        /// Its size.
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// The sum exceeds the budget.
    OverBudget {
        /// Required bytes.
        need: u64,
        /// Available bytes.
        have: u64,
    },
}
