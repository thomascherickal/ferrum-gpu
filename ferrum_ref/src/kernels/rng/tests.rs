use super::*;

#[test]
fn deterministic_and_in_range() {
    let (mut a, mut b) = (Rng::new(7), Rng::new(7));
    for _ in 0..100 {
        let x = a.next_f32();
        assert_eq!(x, b.next_f32());
        assert!((0.0..1.0).contains(&x));
    }
    assert_ne!(Rng::new(7).next_u64(), Rng::new(8).next_u64());
    assert_eq!(Rng::new(0), Rng::new(0));
    assert_ne!(Rng::new(0).next_u64(), 0);
}

#[test]
fn below_stays_in_range_and_covers_it() {
    let mut r = Rng::new(3);
    let mut seen = [0usize; 5];
    for _ in 0..2000 {
        seen[r.below(5)] += 1;
    }
    assert!(seen.iter().all(|&c| c > 300), "{seen:?}");
    assert_eq!(Rng::new(1).below(1), 0);
}

#[test]
#[should_panic(expected = "below(0)")]
fn below_zero_panics() {
    Rng::new(1).below(0);
}

#[test]
fn shuffle_is_a_seeded_permutation() {
    let sorted: Vec<usize> = (0..50).collect();
    let (mut a, mut b, mut c) = (sorted.clone(), sorted.clone(), sorted.clone());
    Rng::new(9).shuffle(&mut a);
    Rng::new(9).shuffle(&mut b);
    Rng::new(10).shuffle(&mut c);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, sorted);
    a.sort();
    assert_eq!(a, sorted, "a permutation: nothing lost or duplicated");
    let mut empty: Vec<u8> = vec![];
    Rng::new(1).shuffle(&mut empty);
    let mut one = vec![5];
    Rng::new(1).shuffle(&mut one);
    assert_eq!(one, vec![5]);
}
