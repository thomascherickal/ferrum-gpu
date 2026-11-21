use super::*;

pub(super) fn build(
    o: &PrepareOptions,
    out: &Path,
    files: Vec<PathBuf>,
) -> Result<PrepareReport, String> {
    let mut splits = [
        Split::create(out, "train")?,
        Split::create(out, "val")?,
        Split::create(out, "test")?,
    ];
    let val_cut = (o.val_frac * 10_000.0).round() as u64;
    let test_cut = val_cut + (o.test_frac * 10_000.0).round() as u64;
    let mut rejected: BTreeMap<&str, u64> = [
        "duplicate",
        "empty",
        "invalid_utf8",
        "too_long",
        "too_short",
    ]
    .map(|k| (k, 0))
    .into();
    let mut lossy_repaired = 0u64;
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let mut uniq: HashSet<char> = HashSet::new();
    let (mut lines, mut words) = (0u64, 0u64);
    let mut inputs = Vec::new();

    for path in &files {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        input::for_each_raw_doc(path, &o.doc_sep, &mut hasher, &mut total, |raw, offset| {
            let text = match std::str::from_utf8(raw) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(_) if o.lossy => {
                    lossy_repaired += 1;
                    String::from_utf8_lossy(raw)
                }
                Err(e) => {
                    return Err(format!(
                        "{}: invalid UTF-8 at byte {} (pass --lossy to replace invalid bytes)",
                        path.display(),
                        offset + e.valid_up_to() as u64
                    ))
                }
            };
            // Every line ending is a newline: CRLF, and the lone CR of classic Mac files.
            let text = if text.contains('\r') {
                std::borrow::Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
            } else {
                text
            };
            let books = if o.strip_gutenberg {
                clean::gutenberg_bodies(&text)
            } else {
                vec![text.as_ref()]
            };
            for book in books {
                let mut doc = clean::strip_invisible(book);
                if o.nfkc {
                    doc = clean::nfkc(&doc);
                }
                let doc = clean::squeeze_spaces(&clean::clean(&doc, o.lowercase));
                let doc = doc.trim();
                let n = doc.chars().count();
                let reason = if n == 0 {
                    Some("empty")
                } else if n < o.min_chars {
                    Some("too_short")
                } else if o.max_chars > 0 && n > o.max_chars {
                    Some("too_long")
                } else {
                    None
                };
                if let Some(r) = reason {
                    *rejected.entry(r).or_default() += 1;
                    continue;
                }
                if o.dedup {
                    let mut h = Sha256::new();
                    h.update(doc.as_bytes());
                    let d = h.digest();
                    if !seen.insert(std::array::from_fn(|i| d[i])) {
                        *rejected.entry("duplicate").or_default() += 1;
                        continue;
                    }
                }
                let which = bucket(o.seed, doc, val_cut, test_cut);
                if which == 0 {
                    uniq.extend(doc.chars());
                    lines += doc.lines().count() as u64;
                    words += doc.split_whitespace().count() as u64;
                }
                splits[which].write(doc)?;
            }
            Ok(())
        })?;
        inputs.push(obj([
            ("path", Json::Str(path.to_string_lossy().into_owned())),
            ("bytes", Json::Num(total)),
            ("sha256", Json::Str(hasher.hex())),
        ]));
    }

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
    let mut manifest = BTreeMap::from([
        (
            "args".to_string(),
            Json::Arr(o.args.iter().cloned().map(Json::Str).collect()),
        ),
        (
            "ferrum_version".into(),
            Json::Str(env!("CARGO_PKG_VERSION").into()),
        ),
        ("inputs".into(), Json::Arr(inputs)),
        ("outputs".into(), outputs),
        (
            "rejected".into(),
            Json::Obj(
                rejected
                    .iter()
                    .map(|(k, n)| (k.to_string(), Json::Num(*n)))
                    .collect(),
            ),
        ),
        ("train_stats".into(), train_stats),
    ]);
    // Only when non-zero, so manifests of clean runs keep their exact bytes.
    if lossy_repaired > 0 {
        manifest.insert("lossy_repaired".into(), Json::Num(lossy_repaired));
    }
    let mpart = part(out, "manifest.json");
    std::fs::write(&mpart, Json::Obj(manifest).pretty() + "\n")
        .map_err(|e| format!("cannot write {}: {e}", mpart.display()))?;
    commit_outputs(out)?;
    Ok(PrepareReport {
        out: out.to_path_buf(),
        documents,
        rejected: rejected
            .into_iter()
            .map(|(k, n)| (k.to_string(), n))
            .collect(),
        lossy_repaired,
    })
}
