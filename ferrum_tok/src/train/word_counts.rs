use super::*;

impl ByteBpeTokenizer {
    pub(super) fn train_from_word_counts(
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

            let touched = merge_in_words(&mut words, &mut where_, pair, new_id);
            apply_deltas(&mut counts, &mut heap, touched);
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
