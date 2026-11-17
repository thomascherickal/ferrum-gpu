//! Reading the machine's memory for the pre-flight. The reads are behind [`Readers`] so tests use
//! fixtures and never run `nvidia-smi` or touch `/proc`; [`System`] is the real implementation and the
//! only part that needs real hardware. What the numbers mean is decided in `ferrum_wgpu::memory`.

use ferrum_wgpu::budget::{parse_nvidia_smi_free, sysfs_vram_mb, Budget};
use ferrum_wgpu::memory::{resolve_memory, MemoryFacts};
use ferrum_wgpu::select::{AdapterSummary, Kind};
use std::path::Path;

/// Where the memory figures come from.
pub trait Readers {
    /// `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`.
    fn nvidia_smi(&self) -> Option<String>;
    /// The text of `/proc/meminfo` (Linux).
    fn meminfo(&self) -> Option<String>;
    /// The text of `sysctl -n hw.memsize` (macOS): total RAM in bytes.
    fn sysctl_memsize(&self) -> Option<String>;
    /// Total VRAM in MiB of each DRM card that reports it (Linux, AMD).
    fn sysfs_vram_mb(&self) -> Vec<u64>;
}

/// The real machine.
pub struct System;

impl Readers for System {
    fn nvidia_smi(&self) -> Option<String> {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn meminfo(&self) -> Option<String> {
        // the kernel's memory report: system information, not a file of this project
        std::fs::read_to_string(Path::new("/proc").join("meminfo")).ok()
    }

    fn sysctl_memsize(&self) -> Option<String> {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn sysfs_vram_mb(&self) -> Vec<u64> {
        sysfs_vram_mb(Path::new("/sys/class/drm"))
    }
}

/// `(MemTotal, MemAvailable)` in bytes from the text of `/proc/meminfo`; either may be missing.
pub fn parse_meminfo(text: &str) -> (Option<u64>, Option<u64>) {
    let kb = |key: &str| -> Option<u64> {
        text.lines()
            .find_map(|l| l.strip_prefix(key))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
            .map(|k| k * 1024)
    };
    (kb("MemTotal:"), kb("MemAvailable:"))
}

/// Does this adapter share the machine's memory (an integrated GPU, or Metal, which is unified)?
pub fn shares_system_memory(a: &AdapterSummary) -> bool {
    a.kind == Kind::Integrated || a.backend == "Metal"
}

fn names_match(smi_name: &str, adapter: &str) -> bool {
    adapter.contains(smi_name) || smi_name.contains(adapter)
}

/// What the readers say about `adapter`'s memory.
pub fn facts_for(a: &AdapterSummary, r: &dyn Readers) -> MemoryFacts {
    let mut f = MemoryFacts::default();
    if !shares_system_memory(a) {
        let smi = r.nvidia_smi().unwrap_or_default();
        if let Some((_, total, free)) = parse_nvidia_smi_free(&smi)
            .into_iter()
            .find(|(n, _, _)| names_match(n, &a.name))
        {
            f.tool_total_mb = total;
            f.tool_free_mb = free;
        } else if a.kind == Kind::Discrete {
            // sysfs cannot be matched to an adapter by name: it is used only when there is one card
            if let [only] = r.sysfs_vram_mb().as_slice() {
                f.tool_total_mb = Some(*only);
            }
        }
    }
    let (total, avail) = r.meminfo().map_or((None, None), |t| parse_meminfo(&t));
    f.ram_total = total.or_else(|| {
        r.sysctl_memsize()
            .and_then(|s| s.trim().parse::<u64>().ok())
    });
    f.ram_available = avail;
    f
}

/// The budget for `adapter` and the facts behind it: the override, else what the readers find.
pub fn budget_for(
    mem_mb: Option<u64>,
    a: &AdapterSummary,
    r: &dyn Readers,
) -> (Option<Budget>, MemoryFacts) {
    let f = facts_for(a, r);
    (resolve_memory(mem_mb, shares_system_memory(a), &f), f)
}

#[cfg(test)]
mod tests;
