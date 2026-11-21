use super::*;

impl CpuTrainer {
    /// A trainer for `w` with `threads` matmul threads (the result does not depend on it).
    pub fn new(w: Gpt2, threads: usize) -> Self {
        let state = AdamState::new(&w);
        CpuTrainer {
            w,
            state,
            threads,
            qat: false,
            tie: false,
            dropout: (0.0, 0),
        }
    }

    /// Train with FFN-hidden dropout at `rate` (clamped to 0.95; 0 or less is off). Each step's
    /// mask is a function of `seed` and the step number, so a resumed run continues the same
    /// masks and two runs with the same seed agree exactly.
    pub fn with_dropout(mut self, rate: f32, seed: u64) -> Self {
        self.dropout = (rate, seed);
        self
    }

    /// Train with the output head tied to the token embedding (`head = tokᵀ`, one shared matrix).
    /// The head is tied at once, and stays tied after every step.
    pub fn with_tie(mut self, on: bool) -> Self {
        self.tie = on;
        if on {
            self.w.tie_head();
        }
        self
    }

    /// Turn quantisation-aware training on or off (off by default).
    pub fn with_qat(mut self, on: bool) -> Self {
        self.qat = on;
        self
    }

    /// The current (master) weights.
    pub fn weights(&self) -> &Gpt2 {
        &self.w
    }
}
