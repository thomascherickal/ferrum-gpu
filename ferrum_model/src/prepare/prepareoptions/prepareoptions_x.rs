use super::*;

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

pub(crate) const OUTPUTS: [&str; 4] = ["train.txt", "val.txt", "test.txt", "manifest.json"];
