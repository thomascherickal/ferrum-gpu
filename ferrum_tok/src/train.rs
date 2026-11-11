// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/train.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! BPE merge learning.
use crate::error::{InferError, Result};
use crate::pretokenize::pretokenize;
use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP, TOK_EOS};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`,
/// scanning left to right.
pub(super) fn merge_pair(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

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

    fn train_from_word_counts(
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

#[cfg(test)]
mod tests {
    use crate::error::InferError;
    use crate::test_support::{docs_corpus, naive_merges, small_tok, CORPUS};
    use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP, TOK_EOS};

    #[test]
    fn merges_compress_the_corpus() {
        let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        assert!(!tok.encode_state().is_empty(), "no merges learned");
        let ids = tok.encode(CORPUS);
        let (n_ids, n_bytes) = (ids.len(), CORPUS.len());
        assert!(n_ids < n_bytes, "{n_ids} ids for {n_bytes} bytes");
    }

    #[test]
    fn vocab_size_is_respected() {
        let tok = ByteBpeTokenizer::train(CORPUS, 260).unwrap();
        assert!(tok.vocab_size() <= 260);
        assert_eq!(tok.vocab_size(), 260); // corpus has ≥4 repeating pairs
    }

    #[test]
    fn training_stops_when_nothing_repeats() {
        // All-distinct bytes: no pair occurs twice, so no merges happen.
        let tok = ByteBpeTokenizer::train("abcdefg", 1000).unwrap();
        assert_eq!(tok.vocab_size(), 257); // 256 bytes + reserved EOS
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn vocab_below_256_errors() {
        assert!(ByteBpeTokenizer::train(CORPUS, 100).is_err());
    }

    #[test]
    fn incremental_training_matches_naive_oracle() {
        // Repeats, overlapping runs ("aaaa"), ties, and multibyte text.
        let corpus = format!(
            "{CORPUS} aaaa aaa abab ababab mississippi banana bandana 🌸🌸 мир мира \n\t {}",
            "the cat sat on the mat with the hat ".repeat(7)
        );
        for vocab in [257, 270, 300, 400] {
            let fast = ByteBpeTokenizer::train(&corpus, vocab).unwrap();
            assert_eq!(fast.merges, naive_merges(&corpus, vocab), "vocab {vocab}");
        }
    }

    #[test]
    fn a_corpus_without_adjacent_pairs_learns_no_merges() {
        for corpus in ["a", "a b"] {
            assert!(naive_merges(corpus, 300).is_empty());
            assert!(ByteBpeTokenizer::train(corpus, 300)
                .unwrap()
                .merges
                .is_empty());
        }
    }

    #[test]
    fn training_is_deterministic() {
        let a = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        let b = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
        assert_eq!(a.encode_state(), b.encode_state());
        assert_eq!(a.encode(CORPUS), b.encode(CORPUS));
    }

    #[test]
    fn multibyte_utf8_merges_safely() {
        // Repeated multi-byte chars: merges may span partial code points
        // internally, but decode must still reassemble valid UTF-8.
        let corpus = "🌸🌸🌸🌸🌸 zen 🌸🌸🌸";
        let tok = ByteBpeTokenizer::train(corpus, 280).unwrap();
        assert_eq!(tok.decode(&tok.encode(corpus)), corpus);
    }

    #[test]
    fn no_merge_spans_whitespace() {
        // "ab ab ab" — without pre-tokenization BPE could learn a token that
        // straddles the space (e.g. "b a"); pre-tokenization forbids it.
        let tok = ByteBpeTokenizer::train("ab ab ab ab ab", 300).unwrap();
        for (id, bytes) in tok.vocab.iter().enumerate() {
            if id >= BASE_VOCAB {
                // GPT-2 chunks may carry one leading space, never an inner one.
                let has_inner_space = bytes.iter().skip(1).any(|&b| (b as char).is_whitespace());
                assert!(
                    !has_inner_space,
                    "merged token {id} ({bytes:?}) spans whitespace"
                );
            }
        }
    }

    #[test]
    fn doc_separator_becomes_eos_when_registered() {
        let corpus = docs_corpus();
        let tok = ByteBpeTokenizer::train_with_docs(&corpus, 300).unwrap();
        assert_eq!(tok.vocab_size(), 300);
        let eos = tok.eos_id().expect("EOS registered");
        assert_eq!(DOC_SEP, TOK_EOS);
        let ids = tok.encode_docs("a.<|endoftext|>b.");
        assert_eq!(ids.iter().filter(|&&i| i == eos).count(), 1);
        assert_eq!(tok.decode(&ids), "a.b.");
        // The separator never leaks into the learned merges.
        assert!(!tok.encode(&corpus).contains(&eos));
        assert!(tok.encode_docs(&corpus).contains(&eos));
    }

    #[test]
    fn train_with_docs_without_separator_equals_plain_training() {
        let corpus = "abc abc abc abd abd abd ".repeat(10);
        let plain = ByteBpeTokenizer::train(&corpus, 262).unwrap();
        let docs = ByteBpeTokenizer::train_with_docs(&corpus, 262).unwrap();
        assert_eq!(plain.encode_state(), docs.encode_state());
        // Vocab below the irreducible minimum still errors through the wrapper.
        assert!(ByteBpeTokenizer::train_with_docs("<|endoftext|>", 256).is_err());
    }

    #[test]
    fn merges_never_cross_pretokenized_chunks() {
        // "ab" appears only across the chunk boundary of "a" + " b"-style splits
        let tok = ByteBpeTokenizer::train(&"a,b ".repeat(30), 300).unwrap();
        let ids = tok.encode("a,b");
        for id in ids {
            let bytes = tok.token_bytes(id).expect("every id has bytes");
            let has_both = bytes.len() > 1 && bytes.contains(&b'a') && bytes.contains(&b',');
            assert!(
                !has_both,
                "token {id} spans letter and punctuation: {bytes:?}"
            );
        }
    }

    #[test]
    fn eos_reservation_edge_cases() {
        // 257 = 256 bytes + EOS and nothing else
        let t = ByteBpeTokenizer::train("aaaa bbbb", 257).unwrap();
        assert_eq!(
            (t.vocab_size(), t.eos_id(), t.num_special()),
            (257, Some(256), 1)
        );
        assert_eq!(t.gguf_merges(), Vec::<String>::new());
        assert_eq!(t.gguf_token_types().last(), Some(&3));
        // EOS lands directly after the last merge, wherever training stopped
        let t = ByteBpeTokenizer::train("ab", 400).unwrap();
        assert_eq!(
            t.vocab_size(),
            257,
            "no repeated pair -> no merges, still EOS"
        );
        assert_eq!(t.eos_id(), Some(256));
        let t = small_tok();
        assert_eq!(t.eos_id(), Some(t.vocab_size() - 1));
        assert!(t.vocab_size() <= 290);
        assert_eq!(
            t.decode(&[t.eos_id().unwrap()]),
            "",
            "EOS decodes to nothing"
        );
        assert_eq!(t.pad_id(), None);
        assert_eq!(t.unk_id(), None);
        assert_eq!(t.bos_id(), None);
        assert!(!t.encode("<|endoftext|>").contains(&t.eos_id().unwrap()));
    }

    #[test]
    fn streaming_reader_and_file_match_in_memory_training() {
        let corpus = format!(
            "{}\n\n{}\n\n{}",
            docs_corpus(),
            "the quick brown fox jumps over the lazy dog.\n".repeat(1600),
            "another line\n\n<|endoftext|>and another doc"
        );
        let mem = ByteBpeTokenizer::train_with_docs(&corpus, 290).unwrap();
        let rdr = ByteBpeTokenizer::train_from_reader(std::io::Cursor::new(corpus.as_bytes()), 290)
            .unwrap();
        assert_eq!(mem.encode_state(), rdr.encode_state());

        let docs: Vec<&str> = corpus.split(DOC_SEP).collect();
        let from_docs =
            ByteBpeTokenizer::train_from_docs(docs.iter().copied().map(Ok), 290).unwrap();
        assert_eq!(mem.encode_state(), from_docs.encode_state());

        let path =
            std::env::temp_dir().join(format!("ferrum_bpe_stream_{}.txt", std::process::id()));
        std::fs::write(&path, &corpus).unwrap();
        let file_tok = ByteBpeTokenizer::train_from_file(&path, 290).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(mem.encode_state(), file_tok.encode_state());

        assert!(ByteBpeTokenizer::train_from_reader(std::io::Cursor::new(b"abc"), 256).is_err());
        assert!(ByteBpeTokenizer::train_from_reader(std::io::Cursor::new(b""), 300).is_ok());
        assert!(ByteBpeTokenizer::train_from_docs(std::iter::once(Ok("abc")), 256).is_err());
        assert!(ByteBpeTokenizer::train_from_docs(
            vec![Err(InferError::Format("boom".into()))],
            300
        )
        .is_err());
        assert!(ByteBpeTokenizer::train_from_file(
            std::path::Path::new("/nonexistent/ferrum/bpe.txt"),
            300
        )
        .is_err());
    }
}
