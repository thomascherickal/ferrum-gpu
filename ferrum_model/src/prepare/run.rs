use super::*;

/// Run the whole pipeline. On any error the previous outputs are left untouched and no `*.part`
/// file remains.
pub fn prepare(o: &PrepareOptions) -> Result<PrepareReport, String> {
    if !(0.0..1.0).contains(&o.val_frac)
        || !(0.0..1.0).contains(&o.test_frac)
        || o.val_frac + o.test_frac >= 1.0
    {
        return Err("--val-frac and --test-frac must be in [0, 1) and sum to less than 1".into());
    }
    let out = Path::new(&o.out);
    let files = expand_inputs(&o.inputs, out)?;
    std::fs::create_dir_all(out)
        .map_err(|e| format!("cannot create directory {}: {e}", out.display()))?;
    // Everything is written to `*.part` and renamed only after every input was read, so a failed
    // run leaves the previous outputs untouched.
    let r = build(o, out, files);
    if r.is_err() {
        for n in OUTPUTS {
            let _ = std::fs::remove_file(part(out, n));
        }
    }
    r
}

/// Move the finished `*.part` files into place. The manifest is the commit marker: an old one is
/// removed first and the new one renamed last, so a `manifest.json` next to the splits always
/// describes a complete set.
// ponytail: the data files are not rolled back; a crash can leave a partial set WITHOUT a manifest,
// never a manifest over a partial set. A staging-directory swap would make the set atomic.
pub(crate) fn commit_outputs(out: &Path) -> Result<(), String> {
    let manifest = out.join("manifest.json");
    match std::fs::remove_file(&manifest) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("cannot remove {}: {e}", manifest.display())),
    }
    for n in OUTPUTS {
        let (from, to) = (part(out, n), out.join(n));
        std::fs::rename(&from, &to)
            .map_err(|e| format!("cannot rename {} to {}: {e}", from.display(), to.display()))?;
    }
    Ok(())
}

/// The split a document belongs to (0 train, 1 val, 2 test): a seeded hash of its text.
pub(crate) fn bucket(seed: u64, doc: &str, val_cut: u64, test_cut: u64) -> usize {
    let mut h = Sha256::new();
    h.update(&seed.to_le_bytes());
    h.update(doc.as_bytes());
    let d = h.digest();
    let b = u64::from_be_bytes(std::array::from_fn(|i| d[i])) % 10_000;
    if b < val_cut {
        1
    } else if b < test_cut {
        2
    } else {
        0
    }
}
