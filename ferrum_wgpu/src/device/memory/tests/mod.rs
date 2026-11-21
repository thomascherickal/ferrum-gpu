use super::*;
use crate::budget::parse_nvidia_smi_free;

const GIB: u64 = 1024 * MIB;

fn facts() -> MemoryFacts {
    MemoryFacts::default()
}

/// The measured MX330 outcomes (capability.md, 2026-11-14), in MiB of memory plan: these loaded and ran,
/// these failed with "GPU out of memory". The card had 1989 MiB free of 2048.
const LOADED: [u64; 6] = [279, 557, 729, 864, 956, 1449];
const FAILED: [u64; 3] = [1721, 1925, 2454];

mod an_integrated_adapter_uses_available_ram_then_half_of_total_then_nothing;
mod nvidia_smi_free_lines_are_read_with_the_same_tolerance_as_the_total_ones;
mod the_headroom_separates_every_plan_that_loaded_from_every_plan_that_failed_on_the_measured_card;
