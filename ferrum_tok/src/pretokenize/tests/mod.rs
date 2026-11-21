use super::{byte_to_unicode, pretokenize};
use crate::ByteBpeTokenizer;

fn pt(text: &str) -> Vec<&str> {
    let got = pretokenize(text);
    assert_eq!(got.concat(), text, "chunks must partition {text:?}");
    assert!(got.iter().all(|c| !c.is_empty()), "empty chunk in {got:?}");
    got
}

mod pretokenize_contractions;
mod pretokenize_is_a_partition_for_mixed_text;
mod pretokenize_partitions_text;
