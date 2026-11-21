use super::*;

/// `export-lmstudio --model F [--repo publisher/name] [--bin P]`.
pub fn export_lmstudio(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    export_lmstudio_with(a, out, 30, std::time::Duration::from_millis(500))
}

/// As [`export_lmstudio`], waiting up to `attempts` x `delay` for LM Studio to list the import.
pub fn export_lmstudio_with(
    a: &Args,
    out: &mut dyn Write,
    attempts: u32,
    delay: std::time::Duration,
) -> Result<(), String> {
    let model = a.need("--model")?;
    let (abs, _) = runnable(model)?;
    let default = format!(
        "ferrum/{}",
        Path::new(model)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let repo = a.get("--repo").map(String::from).unwrap_or(default);
    let (publisher, rest) = repo.split_once('/').unwrap_or(("", ""));
    if publisher.is_empty() || rest.is_empty() || rest.contains('/') {
        return Err(format!("--repo '{repo}' must look like <publisher>/<name>"));
    }
    let tool = find_tool(a.get("--bin"), "FERRUM_LMS", "lms", ".lmstudio/bin/lms")?;
    let (ok, text) = run_tool(
        &tool,
        &[
            "import",
            "-y",
            "-l",
            "--user-repo",
            &repo,
            &abs.to_string_lossy(),
        ],
    )?;
    if !ok {
        return Err(format!("`lms import` failed:\n{}", text.trim()));
    }
    let suffix = format!(
        "{repo}/{}",
        abs.file_name().unwrap_or_default().to_string_lossy()
    );
    for attempt in 0..attempts.max(1) {
        let (ok, listing) = run_tool(&tool, &["ls", "--json"])?;
        if let Some(key) = ok.then(|| model_key_for(&listing, &suffix)).flatten() {
            return say(
                out,
                format!("imported into LM Studio as '{key}' (load it with: lms load {key})"),
            );
        }
        if attempt + 1 < attempts {
            std::thread::sleep(delay);
        }
    }
    Err(format!(
        "imported {repo}, but `lms ls --json` never listed it; open LM Studio and look for {repo}"
    ))
}
