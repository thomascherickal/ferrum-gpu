//! `prepare`: stream raw text, clean it per document, drop duplicates and out-of-range lengths,
//! split train/val/test by a seeded content hash, and write a manifest that pins every input and
//! output by sha256 — a deterministic, DVC-friendly data stage.
//!
//! Provenance: ported from ferrum-cpu `slm_cli/src/prepare/mod.rs` (commit 56a9d3a). Same cleaning
//! order, same split hash and the same `manifest.json` layout, so equal inputs and options give
//! equal `train.txt`/`val.txt`/`test.txt` bytes. Not ported: the progress callback and the stop
//! hook (the CLI has no use for them).

mod clean;
mod input;
mod json;
mod split;

pub use input::expand_inputs;

use crate::sha256::Sha256;
use json::{obj, Json};
use split::{part, Split};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Everything a run needs. The fields mirror the `prepare` flags.
pub struct PrepareOptions {
    /// Files or directories to read.
    pub inputs: Vec<String>,
    /// Output directory.
    pub out: String,
    /// Separator between documents inside a file.
    pub doc_sep: String,
    /// Replace invalid UTF-8 instead of failing.
    pub lossy: bool,
    /// Apply Unicode NFKC.
    pub nfkc: bool,
    /// Lowercase every document.
    pub lowercase: bool,
    /// Keep only the text between Project Gutenberg markers.
    pub strip_gutenberg: bool,
    /// Drop documents shorter than this.
    pub min_chars: usize,
    /// Drop documents longer than this (0 = no limit).
    pub max_chars: usize,
    /// Drop exact duplicate documents.
    pub dedup: bool,
    /// Share of documents for `val.txt`.
    pub val_frac: f64,
    /// Share of documents for `test.txt`.
    pub test_frac: f64,
    /// Seed of the split hash.
    pub seed: u64,
    /// The raw argument list, recorded in the manifest.
    pub args: Vec<String>,
}

/// What a finished run wrote.
#[derive(Clone, Debug)]
pub struct PrepareReport {
    /// The output directory.
    pub out: PathBuf,
    /// Kept documents per split: train, val, test.
    pub documents: [u64; 3],
    /// Rejected documents by reason (`duplicate`, `empty`, `invalid_utf8`, `too_long`, `too_short`).
    pub rejected: BTreeMap<String, u64>,
    /// Kept documents that had invalid bytes replaced (`lossy` only).
    pub lossy_repaired: u64,
}

impl PrepareReport {
    /// The one-line summary the CLI prints.
    pub fn summary(&self, o: &PrepareOptions) -> String {
        let docs: Vec<String> = ["train", "val", "test"]
            .iter()
            .zip(self.documents)
            .map(|(s, n)| format!("{s}={n}"))
            .collect();
        let repaired = if self.lossy_repaired > 0 {
            format!("  lossy_repaired={}", self.lossy_repaired)
        } else {
            String::new()
        };
        let rej: Vec<String> = self
            .rejected
            .iter()
            .filter(|(_, n)| **n != 0)
            .map(|(k, n)| format!("{k}={n}"))
            .collect();
        let rej = if rej.is_empty() {
            "none".into()
        } else {
            rej.join(" ")
        };
        format!(
            "prepared {} -> {}  ({})  rejected {rej}{repaired}",
            o.inputs.join(" "),
            o.out,
            docs.join(" ")
        )
    }
}

const OUTPUTS: [&str; 4] = ["train.txt", "val.txt", "test.txt", "manifest.json"];

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
fn commit_outputs(out: &Path) -> Result<(), String> {
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
fn bucket(seed: u64, doc: &str, val_cut: u64, test_cut: u64) -> usize {
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

fn build(o: &PrepareOptions, out: &Path, files: Vec<PathBuf>) -> Result<PrepareReport, String> {
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

#[cfg(test)]
mod tests;
