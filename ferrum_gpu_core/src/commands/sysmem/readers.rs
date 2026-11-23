use super::*;

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

/// What a tool printed, if it exited successfully.
pub(super) fn stdout_if_ok(out: &std::process::Output) -> Option<String> {
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
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
        stdout_if_ok(&out)
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
        stdout_if_ok(&out)
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

pub(super) fn names_match(smi_name: &str, adapter: &str) -> bool {
    adapter.contains(smi_name) || smi_name.contains(adapter)
}
