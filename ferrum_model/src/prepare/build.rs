use super::ingest::Ingest;
use super::manifest::manifest;
use super::*;

pub(crate) fn build(
    o: &PrepareOptions,
    out: &Path,
    files: Vec<PathBuf>,
) -> Result<PrepareReport, String> {
    let mut run = Ingest::new(o, out)?;
    let mut inputs = Vec::new();
    for path in &files {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        input::for_each_raw_doc(path, &o.doc_sep, &mut hasher, &mut total, |raw, offset| {
            run.raw_doc(path, raw, offset)
        })?;
        inputs.push(obj([
            ("path", Json::Str(path.to_string_lossy().into_owned())),
            ("bytes", Json::Num(total)),
            ("sha256", Json::Str(hasher.hex())),
        ]));
    }
    let Ingest {
        splits,
        rejected,
        lossy_repaired,
        uniq,
        lines,
        words,
        ..
    } = run;
    let [train, val, test] = splits;
    let documents = [train.docs, val.docs, test.docs];
    if train.docs == 0 {
        return Err(format!(
            "train split is empty: {} documents found, need at least 3 (one file is one document \
             unless it contains the separator {:?}; use --doc-sep or add more files)",
            documents.iter().sum::<u64>(),
            o.doc_sep
        ));
    }
    let (train, train_bytes, train_chars) = train.finish()?;
    let train_stats = obj([
        ("bytes", Json::Num(train_bytes)),
        ("chars", Json::Num(train_chars)),
        ("lines", Json::Num(lines)),
        ("unique_chars", Json::Num(uniq.len() as u64)),
        ("words", Json::Num(words)),
    ]);
    let outputs = obj([
        ("test", test.finish()?.0),
        ("train", train),
        ("val", val.finish()?.0),
    ]);
    let mut manifest = manifest(o, inputs, outputs, &rejected, train_stats);
    // Only when non-zero, so manifests of clean runs keep their exact bytes.
    if lossy_repaired > 0 {
        manifest.insert("lossy_repaired".into(), Json::Num(lossy_repaired));
    }
    let mpart = part(out, "manifest.json");
    std::fs::write(&mpart, Json::Obj(manifest).pretty() + "\n")
        .map_err(|e| format!("cannot write {}: {e}", mpart.display()))?;
    commit_outputs(out)?;
    let rejected = rejected
        .into_iter()
        .map(|(k, n)| (k.to_string(), n))
        .collect();
    Ok(PrepareReport {
        out: out.to_path_buf(),
        documents,
        rejected,
        lossy_repaired,
    })
}
