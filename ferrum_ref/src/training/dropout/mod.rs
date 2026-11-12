//! Dropout masks for training (ferrum-cpu's scheme: on the FFN hidden activations, after GELU).
//!
//! Inverted dropout: each unit is zeroed with probability `rate` and the survivors are scaled by
//! `1 / (1 - rate)`, so the expected activation is unchanged and inference needs no correction.
//!
//! A mask is a pure function of `(seed, block, unit index)`: a counter-based hash, not a stream.
//! So the same step gives the same mask however the work is split (threads, batches), a resumed
//! run reproduces the masks of the run it continues, and a GPU kernel can compute any unit's mask
//! on its own, without a shared generator.

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

    /// The multiplier of each of `n` units in block `block`: `0.0` (dropped) or `1 / (1 - rate)`.
    pub fn mask(&self, block: usize, n: usize) -> Vec<f32> {
        let keep = 1.0 / (1.0 - self.rate);
        (0..n)
            .map(|i| {
                if uniform(self.seed, block as u64, i as u64) < self.rate {
                    0.0
                } else {
                    keep
                }
            })
            .collect()
    }
}

/// A well-mixed seed for optimizer step `step` of a run seeded `base`.
pub fn step_seed(base: u64, step: u64) -> u64 {
    splitmix(base ^ splitmix(step.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

/// SplitMix64's finalizer: a bijective 64-bit mixer.
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A uniform value in `[0, 1)` for `(seed, block, index)`: 24 bits, exactly representable in f32.
fn uniform(seed: u64, block: u64, index: u64) -> f32 {
    let h = splitmix(splitmix(seed ^ block.wrapping_mul(0xd6e8_feb8_6659_fd93)) ^ index);
    (h >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod grad_tests;
#[cfg(test)]
mod tests;
