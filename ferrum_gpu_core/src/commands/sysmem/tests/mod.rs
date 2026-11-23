use super::*;
use ferrum_wgpu::budget::Source;

const MIB: u64 = 1024 * 1024;

/// Fixtures instead of the machine.
#[derive(Default)]
struct Fake {
    smi: Option<&'static str>,
    meminfo: Option<&'static str>,
    sysctl: Option<&'static str>,
    sysfs: Vec<u64>,
}

impl Readers for Fake {
    fn nvidia_smi(&self) -> Option<String> {
        self.smi.map(String::from)
    }
    fn meminfo(&self) -> Option<String> {
        self.meminfo.map(String::from)
    }
    fn sysctl_memsize(&self) -> Option<String> {
        self.sysctl.map(String::from)
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

const MEMINFO: &str = "MemTotal:       16094208 kB\nMemFree:         1020000 kB\nMemAvailable:    9313032 kB\nBuffers:          100 kB\n";

mod a_lone_amd_card_falls_back_to_sysfs_but_several_cards_are_not_guessed;
#[cfg(unix)]
mod a_tool_that_failed_gives_no_output;
mod meminfo_gives_total_and_available_in_bytes;
