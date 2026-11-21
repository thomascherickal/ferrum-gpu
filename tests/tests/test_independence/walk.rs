use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

pub(super) fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if name == "target" || name == ".git" || name == ".claude" {
            continue;
        }
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

pub(super) fn files(ext: &str, name: Option<&str>) -> Vec<PathBuf> {
    let mut all = vec![];
    walk(&root(), &mut all);
    all.into_iter()
        .filter(|p| match name {
            Some(n) => p.file_name().is_some_and(|f| f == n),
            None => p.extension().is_some_and(|e| e == ext),
        })
        .collect()
}
