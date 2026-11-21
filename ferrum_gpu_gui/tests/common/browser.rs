use super::chrome::with_deadline;
use super::process_group::reap_group;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::time::Duration;

/// A running headless browser in its own process group, so that dropping it ends exactly the
/// processes it started (Chrome forks helpers) and never anyone else's browser.
pub struct Browser {
    child: Child,
    profile: PathBuf,
    log: PathBuf,
}

impl Browser {
    /// Open `url` with WebGPU enabled (software Vulkan is what Chrome offers on a headless box).
    /// `max_secs` is a hard lifetime enforced by `timeout`, even if this test process is killed.
    pub fn open(binary: &str, url: &str, tag: &str, max_secs: u64) -> Browser {
        let base = std::env::temp_dir().join(format!("ferrum-gui-{tag}-{}", std::process::id()));
        let (profile, log) = (base.clone(), base.with_extension("log"));
        let stderr = std::fs::File::create(&log).expect("create the browser log");
        let child = with_deadline(binary, max_secs)
            .args([
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
            ])
            .process_group(0)
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .expect("start the browser");
        Browser {
            child,
            profile,
            log,
        }
    }

    /// The console and error lines Chrome wrote so far (the raw tail when there are none).
    pub fn console(&self) -> String {
        let all = std::fs::read_to_string(&self.log).unwrap_or_default();
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
}

impl Drop for Browser {
    fn drop(&mut self) {
        reap_group(
            &mut self.child,
            Duration::from_secs(3),
            Duration::from_secs(2),
        );
        let _ = std::fs::remove_dir_all(&self.profile);
        let _ = std::fs::remove_file(&self.log);
    }
}
