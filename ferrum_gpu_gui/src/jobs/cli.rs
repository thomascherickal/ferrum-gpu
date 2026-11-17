//! Finding the `ferrum-gpu` binary the jobs run: an explicit path, then the directory of this
//! executable (a cargo `target/` directory or an install keeps them together), then `PATH`.

use std::path::{Path, PathBuf};

/// The file name of the CLI on this platform.
pub const NAME: &str = if cfg!(windows) {
    "ferrum-gpu.exe"
} else {
    "ferrum-gpu"
};

/// The first existing candidate. `exists` is injected so tests need no real files.
pub fn find(
    explicit: Option<&Path>,
    exe_dir: Option<&Path>,
    path_var: Option<&str>,
    exists: &dyn Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return if exists(p) {
            Ok(p.to_path_buf())
        } else {
            Err(format!("--cli {}: no such file", p.display()))
        };
    }
    let mut tried = vec![];
    if let Some(d) = exe_dir {
        tried.push(d.join(NAME));
    }
    if let Some(v) = path_var {
        tried.extend(std::env::split_paths(v).map(|d| d.join(NAME)));
    }
    tried
        .iter()
        .find(|p| exists(p))
        .cloned()
        .ok_or_else(|| format!("{NAME} was not found next to the GUI or on PATH; pass --cli PATH"))
}

/// [`find`] against the real file system and environment.
pub fn find_here(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf));
    let path_var = std::env::var("PATH").ok();
    find(explicit, exe_dir.as_deref(), path_var.as_deref(), &|p| {
        p.is_file()
    })
}
