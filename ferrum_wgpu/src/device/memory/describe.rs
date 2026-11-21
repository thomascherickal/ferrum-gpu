use super::*;

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
