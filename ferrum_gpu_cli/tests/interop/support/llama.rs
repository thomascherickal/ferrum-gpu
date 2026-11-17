//! Running `llama-completion` the way every interoperability test needs it: greedy, CPU only, one
//! thread, and asked again when llama.cpp gives a truncated answer.

use super::model::{ours, NEW_TOKENS, PROMPT};
use super::proc::with_deadline;
use std::path::Path;
use std::process::Stdio;

/// `llama-completion` on `model`: the prompt then up to [`NEW_TOKENS`] tokens, greedy, CPU only.
/// llama.cpp b11521 sometimes exits 0 having printed nothing, or only the first tokens (seen at
/// `-t 1` while the machine was loaded by a coverage run or a parallel test). An answer that is
/// empty or a strict prefix of what we compute for the same file is asked for again, three
/// attempts in all, and each retry is logged. Any other answer is returned at once, so a real
/// difference still fails.
pub fn completion(dir: &Path, model: &Path) -> String {
    let expected = ours(model);
    let mut text = String::new();
    for attempt in 1..=3 {
        text = completion_once(dir, model);
        if !(text.is_empty() || (expected.starts_with(&text) && text != expected)) {
            return text;
        }
        super::log::step(format!(
            "llama.cpp gave {text:?} for {} (attempt {attempt} of 3), expected {expected:?}",
            model.display()
        ));
    }
    text
}

fn completion_once(dir: &Path, model: &Path) -> String {
    let o = with_deadline(&dir.join("llama-completion"), 120)
        .env("LD_LIBRARY_PATH", dir)
        .args(["-m"])
        .arg(model)
        .args(["-p", PROMPT, "-n", &NEW_TOKENS.to_string()])
        .args([
            "--temp",
            "0",
            "-no-cnv",
            "--no-warmup",
            "-ngl",
            "0",
            "-t",
            "1",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("run llama-completion");
    assert!(
        o.status.success(),
        "llama-completion {}: {}",
        model.display(),
        String::from_utf8_lossy(&o.stderr)
            .lines()
            .filter(|l| l.contains("error") || l.contains("failed"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    String::from_utf8_lossy(&o.stdout).trim_end().to_string()
}
