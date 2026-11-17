//! Listing one folder.

use super::MAX_ENTRIES;
use std::path::{Path, PathBuf};

/// What an entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A folder (or a symlink to one).
    Dir,
    /// A regular file (or a symlink to one).
    File,
    /// Anything else (a socket, a device, a broken link).
    Other,
}

impl Kind {
    /// The word the API reports.
    pub fn word(self) -> &'static str {
        match self {
            Kind::Dir => "dir",
            Kind::File => "file",
            Kind::Other => "other",
        }
    }
}

/// One entry of a listing.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The file name.
    pub name: String,
    /// What it is.
    pub kind: Kind,
    /// Bytes, for a file.
    pub size: u64,
    /// The name starts with a dot.
    pub hidden: bool,
}

/// A folder's contents.
#[derive(Debug, PartialEq)]
pub struct Listing {
    /// The folder, as given.
    pub dir: PathBuf,
    /// Its parent, if it has one.
    pub parent: Option<PathBuf>,
    /// Folders first, then files, each sorted case-insensitively.
    pub entries: Vec<Entry>,
    /// More than [`MAX_ENTRIES`] existed.
    pub truncated: bool,
}

/// List `dir`. An error names the folder and the reason.
pub fn list_dir(dir: &Path) -> Result<Listing, String> {
    list_dir_with(dir, MAX_ENTRIES)
}

/// [`list_dir`] returning at most `max` entries.
pub fn list_dir_with(dir: &Path, max: usize) -> Result<Listing, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut entries = vec![];
    let mut truncated = false;
    for item in rd {
        let Ok(item) = item else { continue };
        if entries.len() >= max {
            truncated = true;
            break;
        }
        let name = item.file_name().to_string_lossy().into_owned();
        // follow symlinks for the kind, so a link to a folder opens like a folder
        let meta = std::fs::metadata(item.path());
        let (kind, size) = match &meta {
            Ok(m) if m.is_dir() => (Kind::Dir, 0),
            Ok(m) if m.is_file() => (Kind::File, m.len()),
            _ => (Kind::Other, 0),
        };
        entries.push(Entry {
            hidden: name.starts_with('.'),
            name,
            kind,
            size,
        });
    }
    entries.sort_by_key(|e| (e.kind != Kind::Dir, e.name.to_lowercase()));
    Ok(Listing {
        dir: dir.to_path_buf(),
        parent: dir
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf),
        entries,
        truncated,
    })
}
