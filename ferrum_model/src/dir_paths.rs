//! Listing a folder as paths: the one place the commands read a directory, so tests can hand them
//! a lister that fails instead of needing a folder the platform refuses to open.
use std::io;
use std::path::{Path, PathBuf};

/// The paths directly inside `dir`, in the order the file system gives them. Failing to open the
/// folder or to read any entry is the error.
pub fn dir_paths(dir: &Path) -> io::Result<Vec<PathBuf>> {
    std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_lists_its_paths_and_a_file_is_not_a_folder() {
        let dir = std::env::temp_dir().join(format!("ferrum_dir_paths_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        std::fs::write(&file, b"a").unwrap();
        assert_eq!(dir_paths(&dir).unwrap(), vec![file.clone()]);
        assert!(dir_paths(&file).is_err());
    }
}
