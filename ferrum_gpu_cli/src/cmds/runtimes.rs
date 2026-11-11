//! `export-ollama` and `export-lmstudio`: hand a GPT-2 GGUF to Ollama or LM Studio. The Modelfile
//! and the `lms` command lines follow what ferrum-cpu uses, so a model made here is registered the
//! same way.

use crate::cmds::say;
use crate::flags::Args;
use ferrum_gpu_gui::json::Json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The Modelfile text for a model at `abs` with `ctx` tokens of context. The path is
/// double-quoted, so callers reject paths with `"` or a newline first.
pub fn modelfile(abs: &Path, ctx: usize) -> String {
    format!(
        "FROM \"{}\"\nTEMPLATE \"{{{{ .Prompt }}}}\"\nPARAMETER stop \"<|endoftext|>\"\nPARAMETER temperature 0.8\nPARAMETER num_ctx {ctx}\n",
        abs.display()
    )
}

/// The absolute path of a GPT-2 model file and its context length; anything else is refused.
fn runnable(model: &str) -> Result<(PathBuf, usize), String> {
    let g = ferrum_gguf::Gguf::parse(
        std::fs::read(model).map_err(|e| format!("cannot read GGUF {model}: {e}"))?,
    )
    .map_err(|e| format!("cannot read GGUF {model}: {e}"))?;
    let arch = g.string("general.architecture").unwrap_or("");
    if arch != "gpt2" {
        return Err(format!(
            "{model}: architecture '{arch}' cannot be exported; only gpt2 models (what `train` writes) run in Ollama and LM Studio from a Modelfile or `lms import`"
        ));
    }
    let ctx = g
        .uint("gpt2.context_length")
        .map_err(|e| format!("{model}: {e}"))?;
    let abs = std::path::absolute(model).map_err(|e| format!("{model}: {e}"))?;
    let s = abs.to_string_lossy();
    if s.contains('"') || s.contains('\n') {
        return Err(format!("{s}: the path contains a double quote or newline, which a Modelfile cannot quote; rename the file"));
    }
    Ok((abs, ctx))
}

/// Find a tool: `--bin`, then the environment variable, then `PATH`, then `fallback` under HOME.
pub fn find_tool(
    flag: Option<&str>,
    env: &str,
    name: &str,
    fallback: &str,
) -> Result<PathBuf, String> {
    find_tool_in(
        flag,
        std::env::var(env).ok().as_deref(),
        env,
        std::env::var_os("PATH").as_deref(),
        name,
        fallback,
    )
}

fn find_tool_in(
    flag: Option<&str>,
    env_val: Option<&str>,
    env: &str,
    path_var: Option<&std::ffi::OsStr>,
    name: &str,
    fallback: &str,
) -> Result<PathBuf, String> {
    if let Some(f) = flag {
        let p = PathBuf::from(f);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!("--bin {f}: no such file"))
        };
    }
    if let Some(v) = env_val.filter(|v| !v.is_empty()) {
        let p = PathBuf::from(v);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!("{env}={v}: no such file"))
        };
    }
    let names = [name.to_string(), format!("{name}.exe")];
    let on_path = path_var.and_then(|pv| {
        std::env::split_paths(pv)
            .flat_map(|d| names.iter().map(move |n| d.join(n)))
            .find(|c| c.is_file())
    });
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| Path::new(&h).join(fallback));
    on_path.or(home.filter(|h| h.is_file())).ok_or_else(|| {
        format!("cannot find `{name}`: pass --bin PATH, set {env}, or put it on PATH")
    })
}

fn run_tool(tool: &Path, args: &[&str]) -> Result<(bool, String), String> {
    let o = Command::new(tool)
        .args(args)
        .output()
        .map_err(|e| format!("{}: {e}", tool.display()))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    Ok((o.status.success(), text))
}

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
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mf = dir.join(format!("{}.Modelfile", name.replace('/', "_")));
    std::fs::write(&mf, &text).map_err(|e| format!("{}: {e}", mf.display()))?;
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

#[cfg(test)]
mod tests;
