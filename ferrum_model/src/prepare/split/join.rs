use super::*;

/// Documents in a file are joined by this line.
pub(super) const JOIN: &str = "\n<|endoftext|>\n";

/// Where `name` is written until the whole run has succeeded.
pub(crate) fn part(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.part"))
}

pub(crate) struct Split {
    pub(super) path: PathBuf,
    pub(super) tmp: PathBuf,
    pub(super) w: BufWriter<Box<dyn Write>>,
    pub(super) hash: Sha256,
    pub(super) bytes: u64,
    pub(crate) docs: u64,
    pub(crate) chars: u64,
}

impl Split {
    pub(crate) fn create(dir: &Path, name: &str) -> Result<Self, String> {
        let path = dir.join(format!("{name}.txt"));
        let tmp = part(dir, &format!("{name}.txt"));
        let file = std::fs::File::create(&tmp)
            .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        Ok(Split::with_writer(path, tmp, Box::new(file)))
    }

    /// A split written to `w`; `tmp` only names it in errors.
    pub(super) fn with_writer(path: PathBuf, tmp: PathBuf, w: Box<dyn Write>) -> Self {
        Split {
            w: BufWriter::new(w),
            path,
            tmp,
            hash: Sha256::new(),
            bytes: 0,
            docs: 0,
            chars: 0,
        }
    }
    pub(crate) fn write(&mut self, doc: &str) -> Result<(), String> {
        if self.docs > 0 {
            self.put(JOIN.as_bytes())?;
        }
        self.put(doc.as_bytes())?;
        self.docs += 1;
        self.chars += doc.chars().count() as u64;
        Ok(())
    }
    pub(super) fn put(&mut self, b: &[u8]) -> Result<(), String> {
        self.w
            .write_all(b)
            .map_err(|e| format!("cannot write {}: {e}", self.tmp.display()))?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
    /// Flush and describe the file for the manifest; the second value is `bytes`, the third `chars`.
    pub(crate) fn finish(mut self) -> Result<(Json, u64, u64), String> {
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
