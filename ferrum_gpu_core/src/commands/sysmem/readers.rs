use super::*;

/// Where the memory figures come from.
pub trait Readers {
    /// `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`.
    fn nvidia_smi(&self) -> Option<String>;
    /// `(total, available)` system RAM in bytes; either may be unknown.
    fn ram_bytes(&self) -> (Option<u64>, Option<u64>);
    /// Total VRAM in MiB of each DRM card that reports it (Linux AMD; empty elsewhere).
    fn sysfs_vram_mb(&self) -> Vec<u64>;
}

/// What a tool printed, if it exited successfully.
pub(super) fn stdout_if_ok(success: bool, stdout: &[u8]) -> Option<String> {
    success.then(|| String::from_utf8_lossy(stdout).into_owned())
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
        stdout_if_ok(out.status.success(), &out.stdout)
    }

    fn ram_bytes(&self) -> (Option<u64>, Option<u64>) {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let known = |bytes: u64| (bytes > 0).then_some(bytes);
        (known(sys.total_memory()), known(sys.available_memory()))
    }

    fn sysfs_vram_mb(&self) -> Vec<u64> {
        sysfs_vram_mb(Path::new("/sys/class/drm"))
    }
}

/// Does this adapter share the machine's memory (an integrated GPU, or Metal, which is unified)?
pub fn shares_system_memory(a: &AdapterSummary) -> bool {
    a.kind == Kind::Integrated || a.backend == "Metal"
}

pub(super) fn names_match(smi_name: &str, adapter: &str) -> bool {
    adapter.contains(smi_name) || smi_name.contains(adapter)
}
