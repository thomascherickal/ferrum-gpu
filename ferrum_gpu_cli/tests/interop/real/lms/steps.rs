use super::*;

/// `ferrum-gpu export-lmstudio` of `model` as `repo` through the `lms` binary `tool`.
pub(super) fn import(model: &Path, repo: &str, tool: &Path) {
    step(format!(
        "import: ferrum-gpu export-lmstudio --model {} --repo {repo} --bin {}",
        model.display(),
        tool.display()
    ));
    let o = ferrum(&[
        "export-lmstudio",
        "--model",
        model.to_str().unwrap(),
        "--repo",
        repo,
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
}

/// Exactly one new entry, and every model of the user's is as it was.
pub(super) fn check_registered(before: &[(String, String)], after: &[(String, String)]) {
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
    for (k, p) in before {
        assert!(
            after.contains(&(k.clone(), p.clone())),
            "a model of the user's changed: {k}"
        );
    }
}

/// The imported file is in the library with the bytes of `original`.
pub(super) fn check_on_disk(landed: &Path, original: &Path) {
    let meta = std::fs::symlink_metadata(landed).expect("the imported file is in the library");
    let how = if meta.file_type().is_symlink() {
        "symlink, `-l`"
    } else {
        "copy"
    };
    step(format!("  on disk: {} ({how})", landed.display()));
    assert_eq!(
        std::fs::read(landed).unwrap(),
        std::fs::read(original).unwrap(),
        "the same bytes"
    );
    step("  the file LM Studio holds is byte-identical to the model we trained");
}
