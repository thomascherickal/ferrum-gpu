//! A real `lms import` into the user's LM Studio library, through `ferrum-gpu export-lmstudio`.
//!
//! This writes under `~/.lmstudio/models`, outside the repository, so it is opt-in
//! (`FERRUM_INTEROP_LMS=1`). It uses a publisher folder of its own (`ferrum-interop-test`), refuses
//! to run if that folder already exists, checks that the user's own models are exactly as they were,
//! and removes only the folder it created.

use crate::skip;
use crate::support::log::{opted_in, step};
use crate::support::model::{ferrum, fixture, ours};
use crate::support::proc::with_deadline;
use crate::support::tools::{llamacpp_dir, lms};
use ferrum_gpu_core::json::Json;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PUBLISHER: &str = "ferrum-interop-test";

/// `(modelKey, path)` of every model `lms ls --json` lists.
fn listing(tool: &Path) -> Vec<(String, String)> {
    let o = with_deadline(tool, 120)
        .args(["ls", "--json"])
        .stdin(Stdio::null())
        .output()
        .expect("run lms ls");
    let text = String::from_utf8_lossy(&o.stdout).into_owned();
    let Some(at) = text.find(['[', '{']) else {
        return vec![];
    };
    let Ok(j) = Json::parse(&text[at..]) else {
        return vec![];
    };
    let items = match &j {
        Json::Arr(a) => a.clone(),
        one => vec![one.clone()],
    };
    items
        .iter()
        .filter_map(|e| {
            Some((
                e.get("modelKey")?.as_str()?.to_string(),
                e.get("path")?.as_str()?.to_string(),
            ))
        })
        .collect()
}

/// Removes the publisher folder this test created, whatever the test does.
struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        match std::fs::remove_dir_all(&self.0) {
            Ok(()) => step(format!("cleanup: removed {}", self.0.display())),
            Err(e) => step(format!(
                "cleanup FAILED for {}: {e} (remove it by hand)",
                self.0.display()
            )),
        }
    }
}

#[test]
fn a_real_lms_import_registers_the_model_and_the_file_runs() {
    if !opted_in(
        "FERRUM_INTEROP_LMS",
        "imports a model into the real LM Studio library under ~/.lmstudio",
    ) {
        return;
    }
    let Some(tool) = lms() else {
        skip!("lms not installed (set FERRUM_LMS)")
    };
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let publisher_dir = home.join(".lmstudio/models").join(PUBLISHER);
    step(format!("lms: {}", tool.display()));
    assert!(
        !publisher_dir.exists(),
        "{} already exists; remove it first (this test only deletes a folder it created)",
        publisher_dir.display()
    );

    let before = listing(&tool);
    step(format!(
        "before: `lms ls --json` lists {} models (the user's own; untouched)",
        before.len()
    ));
    assert!(before.iter().all(|(_, p)| !p.contains(PUBLISHER)));

    let f = fixture();
    let repo = format!("{PUBLISHER}/tiny-gpt2");
    let _cleanup = Cleanup(publisher_dir.clone());
    step(format!(
        "import: ferrum-gpu export-lmstudio --model {} --repo {repo} --bin {}",
        f.f32.display(),
        tool.display()
    ));
    let o = ferrum(&[
        "export-lmstudio",
        "--model",
        f.f32.to_str().unwrap(),
        "--repo",
        &repo,
        "--bin",
        tool.to_str().unwrap(),
    ]);
    let (out, err) = (
        String::from_utf8_lossy(&o.stdout).trim().to_string(),
        String::from_utf8_lossy(&o.stderr).trim().to_string(),
    );
    step(format!("  stdout: {out}"));
    if !err.is_empty() {
        step(format!("  stderr: {err}"));
    }
    assert!(o.status.success(), "export-lmstudio failed: {err}");
    assert!(out.contains("imported into LM Studio as"), "{out}");

    let after = listing(&tool);
    step(format!(
        "after: `lms ls --json` lists {} models",
        after.len()
    ));
    let mine: Vec<_> = after
        .iter()
        .filter(|(_, p)| p.contains(PUBLISHER))
        .collect();
    for (key, path) in &mine {
        step(format!("  registered: modelKey {key}  path {path}"));
    }
    assert_eq!(mine.len(), 1, "exactly one new entry: {after:?}");
    assert_eq!(
        after.len(),
        before.len() + 1,
        "the user's models are exactly as before"
    );
    for (k, p) in &before {
        assert!(
            after.contains(&(k.clone(), p.clone())),
            "a model of the user's changed: {k}"
        );
    }

    let landed = publisher_dir.join("tiny-gpt2").join("f32.gguf");
    let meta = std::fs::symlink_metadata(&landed).expect("the imported file is in the library");
    step(format!(
        "  on disk: {} ({})",
        landed.display(),
        if meta.file_type().is_symlink() {
            "symlink, `-l`"
        } else {
            "copy"
        }
    ));
    assert_eq!(
        std::fs::read(&landed).unwrap(),
        std::fs::read(&f.f32).unwrap(),
        "the same bytes"
    );
    step("  the file LM Studio holds is byte-identical to the model we trained");

    match llamacpp_dir() {
        Some(dir) => {
            step("run: the imported file in llama.cpp (LM Studio's engine), greedy, 1 thread");
            let o = with_deadline(&dir.join("llama-completion"), 120)
                .env("LD_LIBRARY_PATH", &dir)
                .args(["-m"])
                .arg(&landed)
                .args([
                    "-p",
                    crate::support::model::PROMPT,
                    "-n",
                    &crate::support::model::NEW_TOKENS.to_string(),
                ])
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
            assert_eq!(text, ours(&f.f32));
            step("  MATCH");
        }
        None => step("llama.cpp not installed: the file is registered, the engine was not run"),
    }
}
