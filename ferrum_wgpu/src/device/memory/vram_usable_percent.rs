use super::*;

/// Share of a discrete card's free memory a plan may use (measured on one card, see above).
pub const VRAM_USABLE_PERCENT: u64 = 80;
/// Share of the available RAM an integrated or unified-memory adapter may use (an assumption).
pub const UNIFIED_USABLE_PERCENT: u64 = 70;
/// Share of the total VRAM used when only the total is known (no free figure): the display and other
/// programs take part of it.
pub const VRAM_TOTAL_USABLE_PERCENT: u64 = 70;

pub(super) const MIB: u64 = 1024 * 1024;

/// What the readers found out about one adapter's memory; every field may be unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryFacts {
    /// Free dedicated memory, in MiB (for example from `nvidia-smi`).
    pub tool_free_mb: Option<u64>,
    /// Total dedicated memory, in MiB (`nvidia-smi` or sysfs).
    pub tool_total_mb: Option<u64>,
    /// RAM still available, in bytes (Linux `MemAvailable`).
    pub ram_available: Option<u64>,
    /// Total RAM, in bytes.
    pub ram_total: Option<u64>,
}

pub(super) fn percent(bytes: u64, pct: u64) -> u64 {
    // widen first: a budget in the hundreds of gigabytes times 80 must not overflow
    (u128::from(bytes) * u128::from(pct) / 100) as u64
}

/// The budget for an adapter, or `None` when nothing is known (then only the binding limit is checked).
///
/// Order: the `--gpu-mem-mb` override; for an `integrated` (or unified-memory) adapter a share of the
/// available RAM, else half of the total RAM; for a discrete one a share of its free memory, else of its
/// total.
pub fn resolve_memory(
    override_mb: Option<u64>,
    integrated: bool,
    f: &MemoryFacts,
) -> Option<Budget> {
    if let Some(mb) = override_mb {
        return Some(Budget {
            bytes: mb * MIB,
            source: Source::Override,
        });
    }
    if integrated {
        return match (f.ram_available, f.ram_total) {
            (Some(avail), _) => Some(Budget {
                bytes: percent(avail, UNIFIED_USABLE_PERCENT),
                source: Source::Ram,
            }),
            (None, Some(total)) => Some(Budget {
                bytes: total / 2,
                source: Source::HalfRam,
            }),
            (None, None) => None,
        };
    }
    if let Some(free) = f.tool_free_mb {
        return Some(Budget {
            bytes: percent(free * MIB, VRAM_USABLE_PERCENT),
            source: Source::ToolFree,
        });
    }
    f.tool_total_mb.map(|total| Budget {
        bytes: percent(total * MIB, VRAM_TOTAL_USABLE_PERCENT),
        source: Source::Tool,
    })
}
