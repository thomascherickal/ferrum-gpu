//! Launching headless Chrome against the GUI server and collecting what the page reports.
#![allow(dead_code)]

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A Chrome/Chromium binary, if one is installed.
pub fn chrome() -> Option<String> {
    [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    .into_iter()
    .find(|b| {
        Command::new(b)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
    .map(String::from)
}

/// A running headless browser in its own process group, so that dropping it ends exactly the
/// processes it started (Chrome forks helpers) and never anyone else's browser.
pub struct Browser {
    child: Child,
    profile: PathBuf,
    log: PathBuf,
}

impl Browser {
    /// Open `url` with WebGPU enabled (software Vulkan is what Chrome offers on a headless box).
    pub fn open(binary: &str, url: &str, tag: &str) -> Browser {
        let base = std::env::temp_dir().join(format!("ferrum-gui-{tag}-{}", std::process::id()));
        let (profile, log) = (base.clone(), base.with_extension("log"));
        let stderr = std::fs::File::create(&log).expect("create the browser log");
        let child = Command::new(binary)
            .args([
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu-sandbox",
                "--enable-unsafe-webgpu",
                "--enable-features=Vulkan",
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
        // Terminate the whole group the browser leads (its pid is the group id).
        let _ = Command::new("kill")
            .args(["-TERM", &format!("-{}", self.child.id())])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.profile);
        let _ = std::fs::remove_file(&self.log);
    }
}

/// Poll `fetch` (which returns the page's reported JSON text, if any) until it yields or `limit`
/// passes.
pub fn wait_for(limit: Duration, mut fetch: impl FnMut() -> Option<String>) -> Option<String> {
    let t0 = Instant::now();
    while t0.elapsed() < limit {
        if let Some(s) = fetch() {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    None
}
