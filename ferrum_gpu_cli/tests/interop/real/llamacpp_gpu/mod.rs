//! llama.cpp's Vulkan build on each GPU: the same greedy text as ours for every precision, with the
//! verbose log as proof that the layers really were on the GPU (and a CPU control that they were not).
use crate::skip;
use crate::support::log::{grep, opted_in, step};
use crate::support::model::{fixture, ours, NEW_TOKENS, PROMPT};
use crate::support::proc::with_deadline;
use crate::support::tools::llamacpp_vulkan_dir;
use std::path::Path;
use std::process::Stdio;

/// What one verbose `llama-completion` run produced.
struct Run {
    text: String,
    log: String,
}

fn completion(dir: &Path, model: &Path, offload: &[&str]) -> Run {
    let o = with_deadline(dir.join("llama-completion"), 540)
        .env("LD_LIBRARY_PATH", dir)
        .args(["-v", "-m"])
        .arg(model)
        .args(["-p", PROMPT, "-n", &NEW_TOKENS.to_string()])
        .args(["--temp", "0", "-no-cnv", "--no-warmup", "-t", "1"])
        .args(offload)
        .stdin(Stdio::null())
        .output()
        .expect("run llama-completion");
    assert!(
        o.status.success(),
        "llama-completion {offload:?}: {}",
        String::from_utf8_lossy(&o.stderr)
            .lines()
            .last()
            .unwrap_or("")
    );
    Run {
        text: String::from_utf8_lossy(&o.stdout).trim_end().to_string(),
        log: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// `Vulkan0`, `Vulkan1`, ... as listed by `--list-devices`, with their descriptions.
fn devices(dir: &Path) -> Vec<(String, String)> {
    let o = with_deadline(dir.join("llama-completion"), 60)
        .env("LD_LIBRARY_PATH", dir)
        .arg("--list-devices")
        .stdin(Stdio::null())
        .output()
        .expect("list devices");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    text.lines()
        .filter_map(|l| {
            let (name, rest) = l.trim().split_once(": ")?;
            name.starts_with("Vulkan")
                .then(|| (name.to_string(), rest.to_string()))
        })
        .collect()
}

/// `(offloaded, total)` from the `offloaded A/B layers to GPU` line.
fn offloaded(log: &str) -> Option<(u32, u32)> {
    let rest = log.split("offloaded ").nth(1)?;
    let frac = rest.split_whitespace().next()?;
    let (a, b) = frac.split_once('/')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

mod llama_cpp_on_each_gpu_gives_our_text_for_every_precision;
