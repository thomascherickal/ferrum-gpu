//! Dropout masks for training (ferrum-cpu's scheme: on the FFN hidden activations, after GELU).
//!
//! Inverted dropout: each unit is zeroed with probability `rate` and the survivors are scaled by
//! `1 / (1 - rate)`, so the expected activation is unchanged and inference needs no correction.
//!
//! A mask is a pure function of `(seed, block, unit index)`: a counter-based 32-bit hash, not a
//! stream. So the same step gives the same mask however the work is split (threads, batches,
//! micro-batches), a resumed run reproduces the masks of the run it continues, and the GPU
//! computes any unit's mask on its own with the same integer arithmetic (WGSL `u32` wraps like
//! Rust's `wrapping_*`; the final `u32 >> 8` to `f32` is exact), so CPU and GPU masks are
//! bit-identical.

/// Dropout settings for one step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dropout {
    /// Probability of zeroing a unit, in `(0, MAX_RATE]`.
    pub rate: f32,
    /// This step's seed (see [`step_seed`]).
    pub seed: u64,
}

/// The largest rate used (ferrum-cpu clamps to the same value).
pub const MAX_RATE: f32 = 0.95;

impl Dropout {
    /// Dropout at `rate` (clamped to [`MAX_RATE`]), or `None` when `rate` is not positive: no
    /// dropout is the absence of a mask, never a mask of ones.
    pub fn new(rate: f32, seed: u64) -> Option<Dropout> {
        (rate > 0.0).then(|| Dropout {
            rate: rate.min(MAX_RATE),
            seed,
        })
    }

    /// The seed as the two 32-bit words the hash (and the GPU kernel) takes.
    pub fn seed_words(&self) -> (u32, u32) {
        (self.seed as u32, (self.seed >> 32) as u32)
    }

    /// The multiplier of each of `n` units in block `block` starting at global unit `offset`:
    /// `0.0` (dropped) or [`keep_scale`]. `offset + n` must fit in 32 bits.
    pub fn mask_at(&self, block: usize, offset: usize, n: usize) -> Vec<f32> {
        let keep = keep_scale(self.rate);
        let (lo, hi) = self.seed_words();
        (offset..offset + n)
            .map(|i| {
                let u = uniform(lo, hi, block as u32, u32::try_from(i).expect("unit index"));
                if u < self.rate {
                    0.0
                } else {
                    keep
                }
            })
            .collect()
    }

    /// The multiplier of each of the first `n` units in block `block`.
    pub fn mask(&self, block: usize, n: usize) -> Vec<f32> {
        self.mask_at(block, 0, n)
    }
}

/// What a surviving unit is multiplied by.
pub fn keep_scale(rate: f32) -> f32 {
    1.0 / (1.0 - rate)
}

/// A well-mixed seed for optimizer step `step` of a run seeded `base`.
pub fn step_seed(base: u64, step: u64) -> u64 {
    splitmix(base ^ splitmix(step.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

/// SplitMix64's finalizer: a bijective 64-bit mixer (host side only: it makes seeds).
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The "lowbias32" 32-bit mixer. The GPU kernel (`fwd/dropout.wgsl`) is the same function.
pub fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// A uniform value in `[0, 1)` for `(seed, block, index)`: 24 bits, exactly representable in f32.
pub fn uniform(seed_lo: u32, seed_hi: u32, block: u32, index: u32) -> f32 {
    let h = mix(mix(mix(mix(seed_lo) ^ seed_hi) ^ block) ^ index);
    (h >> 8) as f32 / 16_777_216.0
}

#[cfg(test)]
mod grad_tests;
#[cfg(test)]
mod hash_tests;
#[cfg(test)]
mod tests;
