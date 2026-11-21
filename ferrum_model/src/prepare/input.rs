//! Reading the inputs of a corpus run: which files, and their documents streamed in fixed-size
//! chunks.
//!
//! Provenance: ported from ferrum-cpu `slm_cli/src/prepare/input.rs` (commit 56a9d3a).

use crate::sha256::Sha256;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Every regular, non-hidden file under each input (directories walked recursively, FIFOs and
/// devices inside them skipped), sorted by path. A non-regular file named directly is an error
/// (`/dev/zero` would be read forever). Anything under `outdir` is skipped when found by walking
/// and an error when named directly, so a run never re-reads its own outputs.
pub fn expand_inputs(inputs: &[String], outdir: &Path) -> Result<Vec<PathBuf>, String> {
    let outdir = std::fs::canonicalize(outdir).unwrap_or(outdir.to_path_buf());
    let inside_out = |p: &Path| std::fs::canonicalize(p).is_ok_and(|c| c.starts_with(&outdir));
    fn walk(
        p: &Path,
        top: bool,
        inside_out: &dyn Fn(&Path) -> bool,
        out: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        if p.is_dir() {
            if !top && inside_out(p) {
                return Ok(());
            }
            let ctx = |e: std::io::Error| format!("cannot read {}: {e}", p.display());
            for e in std::fs::read_dir(p).map_err(ctx)? {
                let e = e.map_err(ctx)?.path();
                if !e
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                {
                    walk(&e, false, inside_out, out)?;
                }
            }
        } else if p.is_file() {
            if inside_out(p) {
                if top {
                    return Err(format!(
                        "input {} is inside --out (its own output)",
                        p.display()
                    ));
                }
                return Ok(());
            }
            out.push(p.to_path_buf());
        } else if top {
            return Err(format!(
                "input {} is not a regular file or directory",
                p.display()
            ));
        }
        Ok(())
    }
    let mut out = Vec::new();
    for i in inputs {
        let p = Path::new(i);
        if !p.exists() {
            return Err(format!("input {i} does not exist"));
        }
        walk(p, true, &inside_out, &mut out)?;
    }
    out.sort();
    Ok(out)
}

// ponytail: tiny chunk in unit tests so the boundary cases run on 64-byte buffers, not 1 MiB.
const CHUNK: usize = if cfg!(test) { 64 } else { 1 << 20 };

/// Stream `path` as raw documents split on `sep` (byte-exact; an empty separator yields the whole
/// file). Calls `f(bytes, offset)`, hashes every byte into `hasher` and counts them in `total`.
pub fn for_each_raw_doc(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    mut f: impl FnMut(&[u8], u64) -> Result<(), String>,
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
#[cfg(test)]
mod tests;
