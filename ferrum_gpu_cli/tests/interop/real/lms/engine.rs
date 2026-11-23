use super::*;
use crate::support::model::{NEW_TOKENS, PROMPT};

/// Run `landed` in llama.cpp (LM Studio's engine), greedy, one thread; its text is `want`.
pub(super) fn run_in_llamacpp(dir: &Path, landed: &Path, want: &str) {
    step("run: the imported file in llama.cpp (LM Studio's engine), greedy, 1 thread");
    let tokens = NEW_TOKENS.to_string();
    let o = with_deadline(dir.join("llama-completion"), 120)
        .env("LD_LIBRARY_PATH", dir)
        .args(["-m"])
        .arg(landed)
        .args(["-p", PROMPT, "-n", &tokens])
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
        .unwrap();
    let text = String::from_utf8_lossy(&o.stdout).trim_end().to_string();
    step(format!("  llama.cpp text: {text:?}"));
    assert_eq!(text, want);
    step("  MATCH");
}
