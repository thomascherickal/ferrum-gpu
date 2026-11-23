use super::*;

/// `export-ollama --model F --name NAME [--bin P] [--modelfile-only OUT]`.
pub fn export_ollama(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let (abs, ctx) = runnable(a.need("--model")?)?;
    let text = modelfile(&abs, ctx);
    if let Some(p) = a.get("--modelfile-only") {
        std::fs::write(p, &text).map_err(|e| format!("{p}: {e}"))?;
        return say(out, format!("wrote {p}"));
    }
    let name = a.need("--name")?;
    let tool = find_tool(
        a.get("--bin"),
        "FERRUM_OLLAMA",
        "ollama",
        ".local/bin/ollama",
    )?;
    let dir = std::env::temp_dir().join(format!("ferrum_modelfile_{}", std::process::id()));
    let mf = write_modelfile(&dir, name, &text)?;
    let res = run_tool(&tool, &["create", name, "-f", &mf.to_string_lossy()]);
    let _ = std::fs::remove_dir_all(&dir);
    let (ok, output) = res?;
    if !ok {
        let hint = if output.contains("connect") || output.contains("refused") {
            "\nthe Ollama server is not running: start it with `ollama serve`"
        } else {
            ""
        };
        return Err(format!(
            "`ollama create {name}` failed:\n{}{hint}",
            output.trim()
        ));
    }
    say(
        out,
        format!("created Ollama model '{name}' (run it with: ollama run {name})"),
    )
}

/// Write the Modelfile for `name` into `dir` (made if missing) and return its path.
pub(super) fn write_modelfile(dir: &Path, name: &str, text: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mf = dir.join(format!("{}.Modelfile", name.replace('/', "_")));
    std::fs::write(&mf, text).map_err(|e| format!("{}: {e}", mf.display()))?;
    Ok(mf)
}

/// The `modelKey` of the `lms ls --json` entry whose path ends with `suffix` on a `/` boundary.
/// The JSON may follow a banner line, so parsing starts at the first `[` or `{`.
pub fn model_key_for(json: &str, suffix: &str) -> Option<String> {
    let parsed = Json::parse(&json[json.find(['[', '{'])?..]).ok()?;
    let entries = match &parsed {
        Json::Arr(a) => a.as_slice(),
        one => std::slice::from_ref(one),
    };
    entries.iter().find_map(|e| {
        let path = e.get("path")?.as_str()?;
        let hit = path
            .strip_suffix(suffix)
            .is_some_and(|h| h.is_empty() || h.ends_with('/'));
        hit.then(|| e.get("modelKey").and_then(Json::as_str).map(String::from))
            .flatten()
    })
}
