//! Seeded xorshift64* generator: deterministic, no dependencies.

/// A small deterministic random number generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng(u64);

impl Rng {
    /// A generator from `seed` (0 is remapped to a fixed non-zero state).
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `0..n` (`n > 0`), by rejection so the result is unbiased.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        let n = n as u64;
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let r = self.next_u64();
            if r < zone {
                return (r % n) as usize;
            }
        }
    }

    /// Fisher-Yates shuffle in place.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
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
}
