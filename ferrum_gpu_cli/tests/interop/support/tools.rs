//! Finding the external tools. Each lookup returns `None` when the tool is absent.

use std::path::{Path, PathBuf};

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// The directory of an unpacked llama.cpp release whose binaries are in `d` or one level down.
fn release_root(d: &Path) -> Option<PathBuf> {
    let has = |d: &Path| d.join("llama-completion").is_file() && d.join("llama-tokenize").is_file();
    if has(d) {
        return Some(d.to_path_buf());
    }
    std::fs::read_dir(d)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.is_dir() && has(p))
}

/// Newest `~/.local/opt/llama.cpp-*` whose name does (or does not) end in `-vulkan`.
fn newest_under_opt(vulkan: bool) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(home()?.join(".local/opt"))
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                n.starts_with("llama.cpp-") && n.ends_with("-vulkan") == vulkan
            })
        })
        .collect();
    dirs.sort();
    dirs.into_iter().rev().find_map(|d| release_root(&d))
}

/// A directory holding the CPU build's `llama-completion`, `llama-tokenize` and `llama-quantize`:
/// `$FERRUM_LLAMACPP_DIR`, else the newest `~/.local/opt/llama.cpp-*`.
pub fn llamacpp_dir() -> Option<PathBuf> {
    match std::env::var_os("FERRUM_LLAMACPP_DIR") {
        Some(d) => release_root(Path::new(&d)),
        None => newest_under_opt(false),
    }
}

/// The Vulkan build of llama.cpp (it has `libggml-vulkan.so`): `$FERRUM_LLAMACPP_VULKAN_DIR`, else
/// the newest `~/.local/opt/llama.cpp-*-vulkan`.
pub fn llamacpp_vulkan_dir() -> Option<PathBuf> {
    let root = match std::env::var_os("FERRUM_LLAMACPP_VULKAN_DIR") {
        Some(d) => release_root(Path::new(&d)),
        None => newest_under_opt(true),
    }?;
    root.join("libggml-vulkan.so").is_file().then_some(root)
}

/// The `ollama` binary: `$FERRUM_OLLAMA`, else `PATH`, else `~/.local/bin/ollama`.
pub fn ollama() -> Option<PathBuf> {
    on_path_or("FERRUM_OLLAMA", "ollama", ".local/bin/ollama")
}

/// The `lms` CLI of LM Studio: `$FERRUM_LMS`, else `PATH`, else `~/.lmstudio/bin/lms`.
pub fn lms() -> Option<PathBuf> {
    on_path_or("FERRUM_LMS", "lms", ".lmstudio/bin/lms")
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
