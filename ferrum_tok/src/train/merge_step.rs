use super::*;

/// Merge `pair` into `new_id` in every word that holds it; returns the change in each pair's
/// count. `where_` (pair → word indices) gains the pairs that involve the new token.
pub(super) fn merge_in_words(
    words: &mut [(Vec<u32>, i64)],
    where_: &mut HashMap<(u32, u32), Vec<usize>>,
    pair: (u32, u32),
    new_id: u32,
) -> HashMap<(u32, u32), i64> {
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
    touched
}

/// Fold the count changes into `counts` and re-queue every pair that is still live.
pub(super) fn apply_deltas(
    counts: &mut HashMap<(u32, u32), i64>,
    heap: &mut BinaryHeap<(i64, Reverse<(u32, u32)>)>,
    touched: HashMap<(u32, u32), i64>,
) {
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
