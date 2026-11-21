use super::decode::decode;
use super::filters::{digest, length_reject};
use super::*;

/// The running state of one `prepare`: the three split writers, the reject counts and the
/// statistics of the train split.
pub(super) struct Ingest<'a> {
    o: &'a PrepareOptions,
    pub(super) splits: [Split; 3],
    pub(super) rejected: BTreeMap<&'static str, u64>,
    pub(super) lossy_repaired: u64,
    seen: HashSet<[u8; 16]>,
    pub(super) uniq: HashSet<char>,
    pub(super) lines: u64,
    pub(super) words: u64,
    val_cut: u64,
    test_cut: u64,
}

impl<'a> Ingest<'a> {
    pub(super) fn new(o: &'a PrepareOptions, out: &Path) -> Result<Self, String> {
        let splits = [
            Split::create(out, "train")?,
            Split::create(out, "val")?,
            Split::create(out, "test")?,
        ];
        let val_cut = (o.val_frac * 10_000.0).round() as u64;
        let test_cut = val_cut + (o.test_frac * 10_000.0).round() as u64;
        let rejected = [
            "duplicate",
            "empty",
            "invalid_utf8",
            "too_long",
            "too_short",
        ]
        .map(|k| (k, 0))
        .into();
        Ok(Ingest {
            o,
            splits,
            rejected,
            lossy_repaired: 0,
            seen: HashSet::new(),
            uniq: HashSet::new(),
            lines: 0,
            words: 0,
            val_cut,
            test_cut,
        })
    }

    /// One raw document of `path` (at `offset`): decode, normalise line endings, clean, then keep.
    pub(super) fn raw_doc(&mut self, path: &Path, raw: &[u8], offset: u64) -> Result<(), String> {
        let text = decode(raw, path, offset, self.o.lossy, &mut self.lossy_repaired)?;
        let books = if self.o.strip_gutenberg {
            clean::gutenberg_bodies(&text)
        } else {
            vec![text.as_ref()]
        };
        for book in books {
            let mut doc = clean::strip_invisible(book);
            if self.o.nfkc {
                doc = clean::nfkc(&doc);
            }
            let doc = clean::squeeze_spaces(&clean::clean(&doc, self.o.lowercase));
            self.keep(doc.trim())?;
        }
        Ok(())
    }

    fn reject(&mut self, why: &'static str) {
        *self.rejected.entry(why).or_default() += 1;
    }

    /// Length-filter, deduplicate and write one cleaned document to its split.
    fn keep(&mut self, doc: &str) -> Result<(), String> {
        let o = self.o;
        let reason = length_reject(o, doc.chars().count());
        if let Some(r) = reason {
            self.reject(r);
            return Ok(());
        }
        if o.dedup && !self.seen.insert(digest(doc)) {
            self.reject("duplicate");
            return Ok(());
        }
        let which = bucket(o.seed, doc, self.val_cut, self.test_cut);
        if which == 0 {
            self.uniq.extend(doc.chars());
            self.lines += doc.lines().count() as u64;
            self.words += doc.split_whitespace().count() as u64;
        }
        self.splits[which].write(doc)
    }
}
