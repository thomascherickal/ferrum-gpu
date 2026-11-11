//! One output split (`train.txt`, `val.txt`, `test.txt`): a `*.part` file hashed and counted as it
//! is written.
//!
//! Provenance: ported from ferrum-cpu `slm_cli/src/prepare/split.rs` (commit 56a9d3a).

use super::json::{obj, Json};
use crate::sha256::Sha256;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Documents in a file are joined by this line.
const JOIN: &str = "\n<|endoftext|>\n";

/// Where `name` is written until the whole run has succeeded.
pub(super) fn part(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.part"))
}

pub(super) struct Split {
    path: PathBuf,
    tmp: PathBuf,
    w: BufWriter<std::fs::File>,
    hash: Sha256,
    bytes: u64,
    pub(super) docs: u64,
    pub(super) chars: u64,
}

impl Split {
    pub(super) fn create(dir: &Path, name: &str) -> Result<Self, String> {
        let path = dir.join(format!("{name}.txt"));
        let tmp = part(dir, &format!("{name}.txt"));
        let file = std::fs::File::create(&tmp)
            .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        Ok(Split {
            w: BufWriter::new(file),
            path,
            tmp,
            hash: Sha256::new(),
            bytes: 0,
            docs: 0,
            chars: 0,
        })
    }
    pub(super) fn write(&mut self, doc: &str) -> Result<(), String> {
        if self.docs > 0 {
            self.put(JOIN.as_bytes())?;
        }
        self.put(doc.as_bytes())?;
        self.docs += 1;
        self.chars += doc.chars().count() as u64;
        Ok(())
    }
    fn put(&mut self, b: &[u8]) -> Result<(), String> {
        self.w
            .write_all(b)
            .map_err(|e| format!("cannot write {}: {e}", self.tmp.display()))?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
    /// Flush and describe the file for the manifest; the second value is `bytes`, the third `chars`.
    pub(super) fn finish(mut self) -> Result<(Json, u64, u64), String> {
        self.w
            .flush()
            .map_err(|e| format!("cannot write {}: {e}", self.tmp.display()))?;
        let json = obj([
            ("path", Json::Str(self.path.to_string_lossy().into_owned())),
            ("bytes", Json::Num(self.bytes)),
            ("sha256", Json::Str(self.hash.hex())),
            ("documents", Json::Num(self.docs)),
            ("chars", Json::Num(self.chars)),
        ]);
        Ok((json, self.bytes, self.chars))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_split_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_split_joins_documents_and_counts_them() {
        let d = tmp("ok");
        let mut s = Split::create(&d, "train").unwrap();
        s.write("ab").unwrap();
        s.write("cd").unwrap();
        let (json, bytes, chars) = s.finish().unwrap();
        let want = format!("ab{JOIN}cd");
        assert_eq!((bytes as usize, chars), (want.len(), 4));
        assert!(json
            .pretty()
            .contains(&crate::sha256::sha256_hex(want.as_bytes())));
        assert_eq!(
            std::fs::read_to_string(part(&d, "train.txt")).unwrap(),
            want
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn create_names_the_part_file_when_it_cannot_be_made() {
        let e = Split::create(Path::new("/nonexistent-ferrum-dir"), "train")
            .err()
            .unwrap();
        assert!(
            e.starts_with("cannot create ") && e.contains("train.txt.part"),
            "{e}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_failed_write_and_a_failed_flush_name_the_part_file() {
        let d = tmp("full");
        std::os::unix::fs::symlink("/dev/full", part(&d, "train.txt")).unwrap();
        let mut s = Split::create(&d, "train").unwrap();
        let e = s.write(&"x".repeat(1 << 20)).unwrap_err();
        assert!(
            e.starts_with("cannot write ") && e.contains("train.txt.part"),
            "{e}"
        );
        let d2 = tmp("full2");
        std::os::unix::fs::symlink("/dev/full", part(&d2, "train.txt")).unwrap();
        let mut s = Split::create(&d2, "train").unwrap();
        s.write("small").unwrap();
        assert!(s.finish().err().unwrap().starts_with("cannot write "));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }
}
