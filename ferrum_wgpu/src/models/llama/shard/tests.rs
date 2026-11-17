use super::*;

#[test]
fn what_fits_is_one_block() {
    assert_eq!(blocks(10, 4, 40).unwrap(), vec![(0, 10)]);
    assert_eq!(blocks(10, 4, 1 << 40).unwrap(), vec![(0, 10)]);
    assert_eq!(blocks(0, 4, 1).unwrap(), vec![(0, 0)]);
    assert_eq!(blocks(5, 0, 1).unwrap(), vec![(0, 5)]);
}

#[test]
fn what_does_not_fit_is_cut_into_blocks_that_cover_it_exactly() {
    // 10 units of 4 bytes with a 16-byte limit: 4 units per block
    let b = blocks(10, 4, 16).unwrap();
    assert_eq!(b, vec![(0, 4), (4, 4), (8, 2)]);
    for total in 1..60usize {
        for unit in [1u64, 3, 4, 7] {
            for limit in [unit, unit * 2, unit * 5, unit * 13] {
                let b = blocks(total, unit, limit).unwrap();
                assert_eq!(
                    b.iter().map(|x| x.1).sum::<usize>(),
                    total,
                    "{total} {unit} {limit}"
                );
                let mut at = 0;
                for (s, l) in &b {
                    assert_eq!(*s, at, "contiguous");
                    assert!(
                        *l as u64 * unit <= limit.max(total as u64 * unit).min(limit),
                        "{total} {unit} {limit}"
                    );
                    at += l;
                }
            }
        }
    }
}

#[test]
fn a_unit_larger_than_the_limit_is_an_error_that_says_so() {
    let e = blocks(10, 100, 99).unwrap_err().to_string();
    assert!(e.contains("larger than the binding limit of 99"), "{e}");
}

#[test]
fn a_column_block_is_the_matching_columns_of_every_row() {
    // [3, 4]
    let w: Vec<f32> = (0..12).map(|v| v as f32).collect();
    assert_eq!(
        column_block(&w, (3, 4), (1, 2)),
        vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
    );
    assert_eq!(column_block(&w, (3, 4), (0, 4)), w);
    assert_eq!(column_block(&w, (3, 4), (3, 1)), vec![3.0, 7.0, 11.0]);
}

#[test]
fn only_weights_and_the_embedding_can_be_split() {
    for n in [
        "tok_emb",
        "head",
        "blk0.wq",
        "blk11.down",
        "blk3.gate",
        "blk3.up",
        "blk2.wo",
        "blk2.wk",
        "blk2.wv",
    ] {
        assert!(splittable(n), "{n}");
    }
    for n in [
        "blk0.kcache",
        "blk0.vcache",
        "scratch.logits",
        "scratch.probs",
        "rope_cos",
        "out_norm",
        "blk0.attn_norm",
        "scratch.head",
    ] {
        assert!(!splittable(n), "{n}");
    }
}
