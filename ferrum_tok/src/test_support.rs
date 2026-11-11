// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/test_support.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! Shared fixtures for the unit tests of this module tree.
use crate::pretokenize::pretokenize;
use crate::train::merge_pair;
use crate::{ByteBpeTokenizer, BASE_VOCAB, TOK_EOS};
use std::collections::HashMap;

pub(super) const CORPUS: &str = "low lower lowest low low newer newest new news";

/// The pre-incremental trainer (full recount per merge), kept as an oracle.
pub(super) fn naive_merges(corpus: &str, vocab_size: usize) -> Vec<(u32, u32)> {
    let mut chunks: Vec<Vec<u32>> = pretokenize(corpus)
        .iter()
        .map(|c| c.bytes().map(u32::from).collect())
        .collect();
    let mut merges = Vec::new();
    while BASE_VOCAB + merges.len() < vocab_size - 1 {
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        for chunk in &chunks {
            for w in chunk.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
        }
        let Some((pair, count)) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(&p, &c)| (p, c))
        else {
            break;
        };
        if count < 2 {
            break;
        }
        let new_id = (BASE_VOCAB + merges.len()) as u32;
        merges.push(pair);
        for chunk in &mut chunks {
            *chunk = merge_pair(chunk, pair, new_id);
        }
    }
    merges
}

/// A corpus with enough distinct words to always reach the requested vocab.
pub(super) fn docs_corpus() -> String {
    (0..200)
        .map(|i| format!("the cat{i} sat{} on mat{}.<|endoftext|>", i % 13, i % 7))
        .collect::<String>()
        .repeat(2)
}

pub(super) fn small_tok() -> ByteBpeTokenizer {
    ByteBpeTokenizer::train(&"abab cdcd ünï ".repeat(12), 290).unwrap()
}

pub(super) fn trained_tok() -> ByteBpeTokenizer {
    let mut tok = ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap();
    tok.add_special_token(TOK_EOS).unwrap();
    tok
}
