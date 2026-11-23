use super::*;

/// Every regular, non-hidden file under each input (directories walked recursively, FIFOs and
/// devices inside them skipped), sorted by path. A non-regular file named directly is an error
/// (`/dev/zero` would be read forever). Anything under `outdir` is skipped when found by walking
/// and an error when named directly, so a run never re-reads its own outputs.
pub fn expand_inputs(inputs: &[String], outdir: &Path) -> Result<Vec<PathBuf>, String> {
    expand_inputs_with(inputs, outdir, &dir_paths)
}

/// How a folder is listed.
pub type Lister<'a> = &'a dyn Fn(&Path) -> std::io::Result<Vec<PathBuf>>;

/// [`expand_inputs`] listing folders through `list`, so a failing lister can be injected.
pub fn expand_inputs_with(
    inputs: &[String],
    outdir: &Path,
    list: Lister,
) -> Result<Vec<PathBuf>, String> {
    let outdir = std::fs::canonicalize(outdir).unwrap_or(outdir.to_path_buf());
    let inside_out = |p: &Path| std::fs::canonicalize(p).is_ok_and(|c| c.starts_with(&outdir));
    fn walk(
        p: &Path,
        top: bool,
        (inside_out, list): (&dyn Fn(&Path) -> bool, Lister),
        out: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        if p.is_dir() {
            if !top && inside_out(p) {
                return Ok(());
            }
            let entries = list(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
            for e in entries {
                if !e
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                {
                    walk(&e, false, (inside_out, list), out)?;
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
        walk(p, true, (&inside_out, list), &mut out)?;
    }
    out.sort();
    Ok(out)
}

// ponytail: tiny chunk in unit tests so the boundary cases run on 64-byte buffers, not 1 MiB.
pub(super) const CHUNK: usize = if cfg!(test) { 64 } else { 1 << 20 };
