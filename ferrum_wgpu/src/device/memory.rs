//! How much device memory a model may use: the budget the pre-flight checks a plan against.
//!
//! `wgpu` has no portable memory query, so the numbers come from outside (the injected readers in
//! `ferrum_gpu_core`), and this module only decides, purely, what they mean. A budget is always an
//! **estimate** and says where it came from ([`Source`]).
//!
//! The headroom is taken from measurement, not guessed. On the NVIDIA MX330 (2048 MiB total, 1989 MiB
//! free) memory plans of up to 1449 MiB (73% of the free memory) loaded and ran, and plans of 1721 MiB
//! (86%) and above failed with "GPU out of memory" (`CAPABILITY.md`, 2026-11-14): the plan counts the
//! resident buffers, while loading and the driver need room beyond them. So a discrete card's usable
//! budget is [`VRAM_USABLE_PERCENT`] of its **free** memory. Integrated and unified-memory adapters
//! share the machine's RAM with everything else, so their budget is a share of the RAM that is still
//! available ([`UNIFIED_USABLE_PERCENT`], an assumption: it was not measured on unified-memory hardware).

use super::budget::{Budget, Source};

/// Share of a discrete card's free memory a plan may use (measured on one card, see above).
pub const VRAM_USABLE_PERCENT: u64 = 80;
/// Share of the available RAM an integrated or unified-memory adapter may use (an assumption).
pub const UNIFIED_USABLE_PERCENT: u64 = 70;
/// Share of the total VRAM used when only the total is known (no free figure): the display and other
/// programs take part of it.
pub const VRAM_TOTAL_USABLE_PERCENT: u64 = 70;

const MIB: u64 = 1024 * 1024;

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

fn percent(bytes: u64, pct: u64) -> u64 {
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

#[cfg(test)]
mod tests;

/// A sentence saying what a budget is and where it came from, for `info` and the GUI.
pub fn describe(b: &Budget, f: &MemoryFacts) -> String {
    let mib = |x: u64| format!("{:.0} MiB", x as f64 / MIB as f64);
    let have = mib(b.bytes);
    match b.source {
        Source::Override => format!("{have} (set with --gpu-mem-mb)"),
        Source::ToolFree => format!(
            "{have}: {VRAM_USABLE_PERCENT}% of the {} free ({} total); an estimate, the percentage is measured on one card",
            mib(f.tool_free_mb.unwrap_or(0) * MIB),
            mib(f.tool_total_mb.unwrap_or(0) * MIB)
        ),
        Source::Tool => format!(
            "{have}: {VRAM_TOTAL_USABLE_PERCENT}% of the {} total (free memory is not known); an estimate",
            mib(f.tool_total_mb.unwrap_or(0) * MIB)
        ),
        Source::Ram => format!(
            "{have}: {UNIFIED_USABLE_PERCENT}% of the {} of RAM available (shared with the system); an estimate, not measured on unified memory",
            mib(f.ram_available.unwrap_or(0))
        ),
        Source::HalfRam => format!(
            "{have}: half of the {} of RAM (shared with the system); an estimate",
            mib(f.ram_total.unwrap_or(0))
        ),
    }
}
