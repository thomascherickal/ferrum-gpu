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

    /// Learn BPE merges by streaming `reader` (splitting on `\n\n` or [`DOC_SEP`]
    /// boundaries, or flushing after 64 KiB at a safe newline boundary) so a
    /// multi-gigabyte text file can train a tokenizer in bounded memory while
    /// preserving multi-newline runs across lines.
    pub fn train_from_reader<R: std::io::BufRead>(
        mut reader: R,
        vocab_size: usize,
    ) -> Result<Self> {
        if vocab_size <= BASE_VOCAB {
            return Err(InferError::DimMismatch(format!(
                "vocab_size {vocab_size} < {} (byte base vocabulary + EOS)",
                BASE_VOCAB + 1
            )));
        }
        let mut freq: HashMap<String, i64> = HashMap::new();
        let mut total_bytes = 0usize;
        let mut buf = String::new();
        let mut line = String::new();
        let mut add_chunk = |text: &str| {
            for part in text.split(DOC_SEP) {
                for c in pretokenize(part) {
                    if let Some(count) = freq.get_mut(c) {
                        *count += 1;
                    } else {
                        freq.insert(c.to_owned(), 1);
                    }
                }
            }
        };
        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                if !buf.is_empty() {
                    add_chunk(&buf);
                }
                break;
            }
            total_bytes += n;
            buf.push_str(&line);
            // Flush once the buffer exceeds 64 KiB and does not end in whitespace
            // (or when `line` contains `DOC_SEP`), so whitespace runs across lines
            // stay intact until followed by a non-whitespace line.
            if buf.len() >= (1 << 16) && !line.trim_end_matches(['\r', '\n']).is_empty() {
                // Keep the trailing newline in the next buffer so a blank line
                // following this line still groups `\n\n` together.
                let split = buf.len().saturating_sub(usize::from(buf.ends_with('\n')));
                add_chunk(&buf[..split]);
                buf.drain(..split);
            }
        }
        Self::train_from_word_counts(freq, total_bytes, vocab_size)
    }

    /// Learn BPE merges by streaming `path` from disk via [`train_from_reader`](Self::train_from_reader).
    pub fn train_from_file(path: &std::path::Path, vocab_size: usize) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        Self::train_from_reader(std::io::BufReader::with_capacity(1 << 20, f), vocab_size)
    }

    pub(crate) fn train_from_word_counts(
        freq: HashMap<String, i64>,
        _total_bytes: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let mut tok = Self::byte_level();
        // Sort by byte sequence so word indices are deterministic regardless of
        // whether `freq` was populated from one string or many streamed chunks.
        let mut words: Vec<(Vec<u32>, i64)> = freq
            .into_iter()
            .map(|(c, f)| (c.into_bytes().into_iter().map(u32::from).collect(), f))
            .collect();
        words.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        vprintln!(
            "[tokenizer::train] corpus={} bytes, {} distinct chunks, target vocab={}",
            total_bytes,
            words.len(),
            vocab_size
        );

        // Pair counts are computed once, then updated incrementally per merge
        // (only the words containing the merged pair change), instead of
        // recounting the whole corpus on every merge.
        let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (wi, (ids, f)) in words.iter().enumerate() {
            for w in ids.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += f;
                where_.entry((w[0], w[1])).or_default().push(wi);
            }
        }
        // Max-heap on (count, smallest pair) with lazy invalidation: an entry is
        // live only while it still matches `counts`. Ties break to the smallest
        // pair, so training is deterministic regardless of HashMap order.
        let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> =
            counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

        while tok.vocab.len() < vocab_size - 1 {
            let Some((count, Reverse(pair))) = heap.pop() else {
                break;
            };
            if counts.get(&pair) != Some(&count) {
                continue; // stale entry
            }
            if count < 2 {
                break;
            }
            let new_id = tok.vocab.len() as u32;
            let merged = [
                tok.vocab[pair.0 as usize].clone(),
                tok.vocab[pair.1 as usize].clone(),
            ]
            .concat();
            vprintln!(
                "[tokenizer::train] merge {}: {:?}+{:?} → id {} (count {})",
                tok.merges.len(),
                pair.0,
                pair.1,
                new_id,
                count
            );
            tok.vocab.push(merged);
            tok.merges.push(pair);

            let mut touched: HashMap<(u32, u32), i64> = HashMap::new();
            let mut wis = where_.remove(&pair).unwrap_or_default();
            wis.sort_unstable();
            wis.dedup();
            for wi in wis {
                let (ids, f) = &mut words[wi];
                if !ids.windows(2).any(|w| (w[0], w[1]) == pair) {
                    continue; // stale index entry
                }
                for w in ids.windows(2) {
                    *touched.entry((w[0], w[1])).or_insert(0) -= *f;
                }
                *ids = merge_pair(ids, pair, new_id);
                for w in ids.windows(2) {
                    *touched.entry((w[0], w[1])).or_insert(0) += *f;
                    // Only pairs involving the new token are new to this word.
                    if w[0] == new_id || w[1] == new_id {
                        where_.entry((w[0], w[1])).or_default().push(wi);
                    }
                }
            }
            for (p, delta) in touched {
                if delta == 0 {
                    continue;
                }
                let c = counts.entry(p).or_insert(0);
                *c += delta;
                if *c > 0 {
                    heap.push((*c, Reverse(p)));
                } else {
                    counts.remove(&p);
                }
            }
        }
        tok.rebuild_ranks();
        // ponytail: EOS is always reserved so every exported GGUF has a real
        // `eos_token_id`; make it optional only if a no-EOS model is ever needed.
        tok.add_special_token(TOK_EOS)?;
        vprintln!(
            "[tokenizer::train] done: vocab={}, merges={}",
            tok.vocab.len(),
            tok.merges.len()
        );
        Ok(tok)
    }
}
