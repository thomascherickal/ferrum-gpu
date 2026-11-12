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
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_prep_in_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn split(path: &Path, sep: &str) -> (Vec<Vec<u8>>, Vec<u64>, String, u64) {
        let (mut docs, mut offs) = (Vec::new(), Vec::new());
        let (mut h, mut total) = (Sha256::new(), 0);
        for_each_raw_doc(path, sep, &mut h, &mut total, |d, o| {
            docs.push(d.to_vec());
            offs.push(o);
            Ok(())
        })
        .unwrap();
        (docs, offs, h.hex(), total)
    }

    #[test]
    fn a_separator_on_the_chunk_boundary_is_found_once() {
        let sep = "<|endoftext|>";
        let d = tmpdir("straddle");
        let p = d.join("f.txt");
        for back in 1..sep.len() {
            let mut data = vec![b'b'; CHUNK - back];
            data.extend_from_slice(sep.as_bytes());
            data.extend_from_slice(b"tail");
            std::fs::write(&p, &data).unwrap();
            let (docs, offs, hash, total) = split(&p, sep);
            assert_eq!(docs.len(), 2, "back={back}");
            assert_eq!(
                (docs[0].len(), docs[1].as_slice()),
                (CHUNK - back, &b"tail"[..])
            );
            assert_eq!(offs[1], (CHUNK - back + sep.len()) as u64);
            assert_eq!(
                (total, hash),
                (data.len() as u64, crate::sha256::sha256_hex(&data))
            );
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn no_separator_means_one_document_and_an_empty_file_one_empty_document() {
        let d = tmpdir("one");
        let p = d.join("f.txt");
        std::fs::write(&p, vec![b'x'; 2 * CHUNK + 17]).unwrap();
        let (docs, offs, _, total) = split(&p, "");
        assert_eq!(
            (docs.len(), docs[0].len(), offs[0], total),
            (1, 2 * CHUNK + 17, 0, total)
        );
        std::fs::write(&p, b"").unwrap();
        assert_eq!(split(&p, "<|endoftext|>").0, vec![Vec::<u8>::new()]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn callback_errors_and_missing_files_propagate() {
        let d = tmpdir("err");
        let p = d.join("f.txt");
        std::fs::write(&p, b"a|b").unwrap();
        let (mut h, mut t) = (Sha256::new(), 0);
        let e = for_each_raw_doc(&p, "|", &mut h, &mut t, |_, _| Err("stop".into())).unwrap_err();
        assert_eq!(e, "stop");
        let e = for_each_raw_doc(&d.join("nope"), "", &mut h, &mut t, |_, _| Ok(())).unwrap_err();
        assert!(e.contains("cannot open"), "{e}");
        // A directory opens on Linux but cannot be read: a clean error.
        assert!(for_each_raw_doc(&d, "", &mut h, &mut t, |_, _| Ok(())).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn expand_inputs_walks_skips_hidden_and_own_outputs() {
        let d = tmpdir("walk");
        let out = d.join("out");
        std::fs::create_dir_all(out.join("nested")).unwrap();
        std::fs::write(d.join("a.txt"), b"a").unwrap();
        std::fs::write(d.join(".hidden"), b"h").unwrap();
        std::fs::write(out.join("train.txt"), b"t").unwrap();
        let dir = d.to_string_lossy().to_string();
        assert_eq!(
            expand_inputs(std::slice::from_ref(&dir), &out).unwrap(),
            vec![d.join("a.txt")]
        );
        assert!(expand_inputs(&[out.to_string_lossy().into()], &out)
            .unwrap()
            .is_empty());
        let e = expand_inputs(&[out.join("train.txt").to_string_lossy().into()], &out).unwrap_err();
        assert!(e.contains("inside --out"), "{e}");
        let e = expand_inputs(&["/nonexistent-ferrum".into()], &out).unwrap_err();
        assert!(e.contains("does not exist"), "{e}");
        // Not a regular file: refused when named, skipped inside a directory.
        let e = expand_inputs(&["/dev/zero".into()], &out).unwrap_err();
        assert!(e.contains("not a regular file"), "{e}");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(d.join("missing"), d.join("dangling")).unwrap();
            assert_eq!(expand_inputs(&[dir], &out).unwrap(), vec![d.join("a.txt")]);
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
