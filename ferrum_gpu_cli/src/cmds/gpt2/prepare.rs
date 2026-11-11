//! `prepare`: clean, deduplicate and split raw text into `train.txt`, `val.txt`, `test.txt` and a
//! hash manifest. The work is `ferrum_model::prepare`; this parses the flags and prints the summary.

use crate::cmds::say;
use crate::flags::{many, switch, val, Args, Flag};
use ferrum_model::prepare::{prepare, PrepareOptions};
use ferrum_tok::DOC_SEP;
use std::io::Write;

/// The `prepare` flags (ferrum-cpu's spellings).
pub const PREPARE_FLAGS: [Flag; 13] = [
    many("--input"),
    val("--out"),
    val("--doc-sep"),
    switch("--lossy"),
    switch("--nfkc"),
    switch("--lowercase"),
    switch("--strip-gutenberg"),
    val("--min-chars"),
    val("--max-chars"),
    switch("--dedup"),
    val("--val-frac"),
    val("--test-frac"),
    val("--seed"),
];

/// `prepare --input FILE|DIR [--input ...] --out DIR [options]`.
pub fn run(a: &Args, raw: &[String], out: &mut dyn Write) -> Result<(), String> {
    let usage = "usage: ferrum-gpu prepare --input <file|dir> [--input ...] --out <dir> [options]";
    if a.all("--input").is_empty() {
        return Err(usage.into());
    }
    let dest = a.get("--out").ok_or(usage)?;
    let o = PrepareOptions {
        inputs: a.all("--input").to_vec(),
        out: dest.to_string(),
        doc_sep: a.get("--doc-sep").unwrap_or(DOC_SEP).to_string(),
        lossy: a.has("--lossy"),
        nfkc: a.has("--nfkc"),
        lowercase: a.has("--lowercase"),
        strip_gutenberg: a.has("--strip-gutenberg"),
        min_chars: a.num("--min-chars", 1)?,
        max_chars: a.num("--max-chars", 0)?,
        dedup: a.has("--dedup"),
        val_frac: a.num("--val-frac", 0.05)?,
        test_frac: a.num("--test-frac", 0.05)?,
        seed: a.num("--seed", 42)?,
        args: raw.to_vec(),
    };
    let report = prepare(&o)?;
    say(out, report.summary(&o))
}
