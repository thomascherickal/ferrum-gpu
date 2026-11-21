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
