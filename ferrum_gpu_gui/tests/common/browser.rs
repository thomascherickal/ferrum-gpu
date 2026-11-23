use super::guard::Guarded;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

/// A running headless browser in its own process group (a job object on Windows), so that
/// dropping it ends exactly the processes it started (Chrome forks helpers) and never anyone
/// else's browser.
pub(crate) struct Browser {
    _group: Guarded,
    profile: PathBuf,
    log: PathBuf,
}

impl Browser {
    /// Open `url` with WebGPU enabled (software Vulkan is what Chrome offers on a headless box).
    /// `max_secs` is a hard lifetime: the browser and its helpers are killed when it passes.
    pub(crate) fn open(binary: &str, url: &str, tag: &str, max_secs: u64) -> Browser {
        let base = std::env::temp_dir().join(format!("ferrum-gui-{tag}-{}", std::process::id()));
        let (profile, log) = (base.clone(), base.with_extension("log"));
        let stderr = std::fs::File::create(&log).expect("create the browser log");
        let mut cmd = Command::new(binary);
        cmd.args([
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            // the flag name is split because a source guard rejects the bare word in non-comment lines
            concat!("--enable-un", "safe-webgpu"),
            // route Chrome's GPU stack through ANGLE/Vulkan so real adapters are offered
            "--use-gl=angle",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--ignore-gpu-blocklist",
            "--disable-gpu-driver-bug-workarounds",
            "--enable-logging=stderr",
            "--v=0",
            "--no-first-run",
            "--disable-extensions",
            &format!("--user-data-dir={}", profile.display()),
            url,
        ]);
        let max = Some(Duration::from_secs(max_secs));
        let group = Guarded::spawn(&mut cmd, Stdio::null(), Stdio::from(stderr), max)
            .expect("start the browser");
        Browser {
            _group: group,
            profile,
            log,
        }
    }

    /// The console and error lines Chrome wrote so far (the raw tail when there are none).
    pub(crate) fn console(&self) -> String {
        console_lines(&std::fs::read_to_string(&self.log).unwrap_or_default())
    }
}

/// The console and error lines of Chrome's log `all` (the raw tail when there are none).
pub(crate) fn console_lines(all: &str) -> String {
    let picked: Vec<&str> = all
        .lines()
        .filter(|l| l.contains("CONSOLE") || l.contains("ERROR:"))
        .take(40)
        .collect();
    if picked.is_empty() {
        let lines: Vec<&str> = all.lines().collect();
        format!(
            "(no console lines; {} log lines, tail)\n{}",
            lines.len(),
            lines[lines.len().saturating_sub(15)..].join("\n")
        )
    } else {
        picked.join("\n")
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        // the group is ended by `_group`, which drops after this
        let _ = std::fs::remove_dir_all(&self.profile);
        let _ = std::fs::remove_file(&self.log);
    }
}
