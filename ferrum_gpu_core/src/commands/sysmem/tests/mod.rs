use super::*;
use ferrum_wgpu::budget::Source;

const MIB: u64 = 1024 * 1024;

/// Fixtures instead of the machine.
#[derive(Default)]
struct Fake {
    smi: Option<&'static str>,
    ram: (Option<u64>, Option<u64>),
    sysfs: Vec<u64>,
}

impl Readers for Fake {
    fn nvidia_smi(&self) -> Option<String> {
        self.smi.map(String::from)
    }
    fn ram_bytes(&self) -> (Option<u64>, Option<u64>) {
        self.ram
    }
    fn sysfs_vram_mb(&self) -> Vec<u64> {
        self.sysfs.clone()
    }
}

fn adapter(name: &str, kind: Kind, backend: &str) -> AdapterSummary {
    AdapterSummary {
        name: name.into(),
        backend: backend.into(),
        kind,
        max_buffer: 1 << 32,
        max_binding: 1 << 30,
        max_workgroup_invocations: 256,
        features: Default::default(),
    }
}

const RAM: (Option<u64>, Option<u64>) = (Some(16094208 * 1024), Some(9313032 * 1024));

mod a_lone_amd_card_falls_back_to_sysfs_but_several_cards_are_not_guessed;
mod a_tool_that_failed_gives_no_output;
mod the_adapter_kinds_decide_who_shares_ram;
