use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

pub(super) fn have_jq() -> bool {
    let ok = Command::new("jq")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("SKIP: jq is not installed");
    }
    ok
}

pub(super) fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gpu_ctx_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A delegate that proves it received the JSON: prints "DELEGATE:<input length>".
pub(super) fn stub_delegate(dir: &Path) -> PathBuf {
    let p = dir.join("delegate.sh");
    std::fs::write(
        &p,
        "#!/usr/bin/env bash\ninput=$(cat)\nprintf 'DELEGATE:%s' \"${#input}\"\n",
    )
    .unwrap();
    p
}

pub(super) fn write_context(project: &Path, used: f64, age_secs: u64) {
    let dir = project.join(".superpowers");
    std::fs::create_dir_all(&dir).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    std::fs::write(
        dir.join("context.json"),
        format!(
            "{{\"context_used_percentage\": {used}, \"written_at\": {}}}",
            now - age_secs
        ),
    )
    .unwrap();
}

pub(super) fn read_used(project: &Path) -> f64 {
    let text = std::fs::read_to_string(project.join(".superpowers/context.json")).unwrap();
    let v: f64 = text
        .split("\"context_used_percentage\":")
        .nth(1)
        .unwrap()
        .split([',', '}'])
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    v
}
