use super::*;
use std::cell::Cell;

/// Deterministic random parameter vectors for test and fixture models: every draw uses the next
/// seed after `seed * 1000`, so a model built in the same order always gets the same weights.
pub(crate) struct SeededInit(Cell<u64>);

impl SeededInit {
    pub(crate) fn new(seed: u64) -> Self {
        SeededInit(Cell::new(seed.max(1) * 1000))
    }

    fn draw(&self, n: usize) -> Vec<f32> {
        self.0.set(self.0.get() + 1);
        rand_vec(n, self.0.get())
    }

    /// `n` values in `[-scale, scale)`.
    pub(crate) fn mat(&self, n: usize, scale: f32) -> Vec<f32> {
        self.draw(n).into_iter().map(|v| v * scale).collect()
    }

    /// `n` norm gains near 1 (`1 + 0.1 * u`, `u` in `[-1, 1)`).
    pub(crate) fn gain(&self, n: usize) -> Vec<f32> {
        self.draw(n).into_iter().map(|v| 1.0 + 0.1 * v).collect()
    }
}
