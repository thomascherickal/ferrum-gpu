//! Seeded xorshift64* generator: deterministic, no dependencies.

#[cfg(test)]
mod rejection_tests {
    use super::*;

    #[test]
    fn below_rejects_draws_in_the_biased_tail() {
        // n = 2^63 + 1 leaves a zone of just over half the u64 range, so about half of all raw
        // draws are rejected; 200 draws hit the rejection branch with probability 1 - 2^-200.
        let n = (1usize << 63) + 1;
        let mut r = Rng::new(11);
        for _ in 0..200 {
            assert!(r.below(n) < n);
        }
    }
}
mod generator;
pub use generator::*;

#[cfg(test)]
mod tests;
