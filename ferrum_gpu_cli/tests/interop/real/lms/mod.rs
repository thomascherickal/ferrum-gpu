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

mod a_real_lms_import_registers_the_model_and_the_file_runs;
