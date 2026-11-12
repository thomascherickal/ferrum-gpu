//! The 32-bit hash and the offset/seed plumbing the GPU kernel depends on.

use super::*;

#[test]
fn a_mask_at_an_offset_is_the_slice_of_the_long_mask() {
    // micro-batches use the global unit index, so splitting a batch keeps its mask
    let d = Dropout::new(0.3, 77).unwrap();
    let whole = d.mask(1, 1000);
    assert_eq!(d.mask_at(1, 0, 400), whole[..400]);
    assert_eq!(d.mask_at(1, 400, 600), whole[400..]);
    assert_eq!(d.mask_at(1, 999, 1), whole[999..]);
}

#[test]
fn the_seed_words_reassemble_the_seed() {
    let d = Dropout::new(0.1, 0x0123_4567_89ab_cdef).unwrap();
    let (lo, hi) = d.seed_words();
    assert_eq!((u64::from(hi) << 32) | u64::from(lo), d.seed);
    assert_eq!((lo, hi), (0x89ab_cdef, 0x0123_4567));
}

#[test]
fn the_mixer_is_a_bijection_that_moves_every_input() {
    assert_eq!(mix(0), 0, "lowbias32 fixes zero");
    // distinct on a block of inputs (a bijection cannot collide)
    let mut seen: Vec<u32> = (1..5000u32).map(mix).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 4999);
    assert!((1..5000u32).all(|x| mix(x) != x));
}

#[test]
fn the_keep_scale_is_one_over_the_keep_probability() {
    assert_eq!(keep_scale(0.5), 2.0);
    assert_eq!(keep_scale(0.0), 1.0);
    assert!((keep_scale(0.25) - 4.0 / 3.0).abs() < 1e-6);
}

#[test]
fn nearby_units_are_not_visibly_correlated() {
    // adjacent units, and the same unit in adjacent blocks, drop independently
    let d = Dropout::new(0.5, 3).unwrap();
    let m = d.mask(0, 20_000);
    let both = m.windows(2).filter(|w| w[0] == 0.0 && w[1] == 0.0).count() as f32 / 19_999.0;
    assert!(
        (both - 0.25).abs() < 0.02,
        "adjacent pairs both dropped: {both}"
    );
    let n = d.mask(1, 20_000);
    let same = m
        .iter()
        .zip(&n)
        .filter(|(a, b)| (**a == 0.0) == (**b == 0.0))
        .count() as f32
        / 20_000.0;
    assert!((same - 0.5).abs() < 0.02, "agreement across blocks: {same}");
}
