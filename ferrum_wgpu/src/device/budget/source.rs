use super::*;

/// Where a budget number came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `--gpu-mem-mb` / `FERRUM_GPU_MEM_MB`.
    Override,
    /// A percentage of the total VRAM `nvidia-smi` or sysfs reports (no free figure was available).
    Tool,
    /// A percentage of the **free** VRAM a tool reports (see `memory`).
    ToolFree,
    /// A percentage of the RAM still available, for integrated and unified-memory adapters.
    Ram,
    /// Half of the total system RAM, when the available figure is unknown.
    HalfRam,
}

/// A memory budget in bytes with its provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Bytes available for buffers.
    pub bytes: u64,
    /// Where it came from.
    pub source: Source,
}

pub(super) const MIB: u64 = 1024 * 1024;

/// `name, total_mib, used_mib` lines (nvidia-smi csv, no header, no units) to `(name, total_mib)`.
/// The name is split from the right so commas inside it survive; junk lines are skipped and
/// `[N/A]` totals give `None`.
pub fn parse_nvidia_smi(out: &str) -> Vec<(String, Option<u64>)> {
    out.lines()
        .filter_map(|line| {
            let mut it = line.trim().rsplitn(3, ',');
            let _used = it.next()?;
            let total = it.next()?.trim();
            let name = it.next()?.trim();
            let mb = match total {
                "[N/A]" | "N/A" => None,
                t => Some(t.parse::<u64>().ok()?),
            };
            (!name.is_empty()).then(|| (name.to_string(), mb))
        })
        .collect()
}

/// `name, total_mib, free_mib` lines (nvidia-smi csv, no header, no units, query
/// `name,memory.total,memory.free`) to `(name, total, free)`. Same tolerance as [`parse_nvidia_smi`]:
/// junk lines are skipped and `[N/A]` fields give `None`.
pub fn parse_nvidia_smi_free(out: &str) -> Vec<(String, Option<u64>, Option<u64>)> {
    let num = |s: &str| match s.trim() {
        "[N/A]" | "N/A" => Some(None),
        t => t.parse::<u64>().ok().map(Some),
    };
    out.lines()
        .filter_map(|line| {
            let mut it = line.trim().rsplitn(3, ',');
            let free = num(it.next()?)?;
            let total = num(it.next()?)?;
            let name = it.next()?.trim();
            (!name.is_empty()).then(|| (name.to_string(), total, free))
        })
        .collect()
}

/// sysfs `mem_info_vram_total` (bytes) to MiB.
pub fn parse_vram_mb(s: &str) -> Option<u64> {
    s.trim().parse::<u64>().ok().map(|b| b / MIB)
}

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
