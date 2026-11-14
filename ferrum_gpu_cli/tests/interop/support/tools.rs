//! Finding the external tools. Each lookup returns `None` when the tool is absent.

use std::path::{Path, PathBuf};

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// A directory holding `llama-completion`, `llama-tokenize` and `llama-quantize`: `$FERRUM_LLAMACPP_DIR`,
/// else the newest `~/.local/opt/llama.cpp-*`.
pub fn llamacpp_dir() -> Option<PathBuf> {
    let has = |d: &Path| d.join("llama-completion").is_file() && d.join("llama-tokenize").is_file();
    if let Some(d) = std::env::var_os("FERRUM_LLAMACPP_DIR").map(PathBuf::from) {
        return has(&d).then_some(d);
    }
    let opt = home()?.join(".local/opt");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(opt)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("llama.cpp-"))
        })
        .collect();
    dirs.sort();
    dirs.into_iter().rev().find(|d| has(d))
}

/// The `ollama` binary: `$FERRUM_OLLAMA`, else `PATH`, else `~/.local/bin/ollama`.
pub fn ollama() -> Option<PathBuf> {
    on_path_or("FERRUM_OLLAMA", "ollama", ".local/bin/ollama")
}

fn on_path_or(env: &str, name: &str, fallback: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env).map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let on_path = std::env::var_os("PATH").and_then(|pv| {
        std::env::split_paths(&pv)
            .map(|d| d.join(name))
            .find(|c| c.is_file())
    });
    on_path.or_else(|| home().map(|h| h.join(fallback)).filter(|p| p.is_file()))
}
