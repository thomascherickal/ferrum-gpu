use super::*;

/// Stream `path` as raw documents split on `sep` (byte-exact; an empty separator yields the whole
/// file). Calls `f(bytes, offset)`, hashes every byte into `hasher` and counts them in `total`.
pub(crate) fn for_each_raw_doc(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    mut f: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<(), String> {
    split_raw_docs(path, sep, hasher, total, &mut f)
}

type DocCallback<'a> = dyn FnMut(&[u8], u64) -> Result<(), String> + 'a;

/// [`for_each_raw_doc`] with the callback behind a pointer, so the body is compiled once.
fn split_raw_docs(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    f: &mut DocCallback<'_>,
) -> Result<(), String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let sep = sep.as_bytes();
    let mut doc: Vec<u8> = Vec::new();
    let mut doc_start = 0u64;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
        };
        hasher.update(&buf[..n]);
        *total += n as u64;
        let old_len = doc.len();
        doc.extend_from_slice(&buf[..n]);
        if sep.is_empty() {
            continue;
        }
        // A separator can only complete inside the new bytes (it may start before them).
        let mut from = old_len.saturating_sub(sep.len() - 1);
        let mut consumed = 0;
        while let Some(i) = doc[from..]
            .windows(sep.len())
            .position(|w| w == sep)
            .map(|i| from + i)
        {
            f(&doc[consumed..i], doc_start)?;
            doc_start += (i + sep.len() - consumed) as u64;
            consumed = i + sep.len();
            from = consumed;
        }
        doc.drain(..consumed);
    }
    f(&doc, doc_start)
}
