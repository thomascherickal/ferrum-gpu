use super::*;

impl ByteBpeTokenizer {
    /// Learn up to `vocab_size − 257` merges from `corpus`, then reserve one id
    /// for [`TOK_EOS`] (so the result has at most `vocab_size` tokens).
    ///
    /// The corpus is pre-tokenized like llama.cpp's `gpt-2` splitter so merges
    /// stay within words; training then greedily merges the most frequent
    /// adjacent pair. Stops early when no adjacent pair occurs at least twice.
    /// `vocab_size` below 257 is an error (256 bytes + EOS are irreducible).
    pub fn train(corpus: &str, vocab_size: usize) -> Result<Self> {
        Self::train_from_docs(std::iter::once(Ok(corpus)), vocab_size)
    }

    /// [`train`](Self::train) that understands document separators: they are
    /// removed before learning merges, and `train` already reserves [`TOK_EOS`]
    /// (the last id), so the total vocabulary is still `vocab_size`. Identical
    /// to `train` when the corpus has no separator.
    pub fn train_with_docs(corpus: &str, vocab_size: usize) -> Result<Self> {
        Self::train(&corpus.replace(DOC_SEP, ""), vocab_size)
    }

    /// Learn BPE merges from a stream of document texts without holding the
    /// whole corpus in memory. Pre-tokenized chunks never merge across each
    /// other or across document boundaries, so only the frequency table of
    /// distinct chunks is kept in RAM.
    pub fn train_from_docs<'a, I>(docs: I, vocab_size: usize) -> Result<Self>
    where
        I: IntoIterator<Item = Result<&'a str>>,
    {
        if vocab_size <= BASE_VOCAB {
            return Err(InferError::DimMismatch(format!(
                "vocab_size {vocab_size} < {} (byte base vocabulary + EOS)",
                BASE_VOCAB + 1
            )));
        }
        let mut freq: HashMap<String, i64> = HashMap::new();
        let mut total_bytes = 0usize;
        for doc in docs {
            let doc = doc?;
            total_bytes += doc.len();
            for part in doc.split(DOC_SEP) {
                for c in pretokenize(part) {
                    if let Some(n) = freq.get_mut(c) {
                        *n += 1;
                    } else {
                        freq.insert(c.to_owned(), 1);
                    }
                }
            }
        }
        Self::train_from_word_counts(freq, total_bytes, vocab_size)
    }
}
