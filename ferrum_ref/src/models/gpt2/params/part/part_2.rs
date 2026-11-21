use super::*;

impl Gpt2 {
    /// A model of the same shape with every parameter zero (a gradient or moment container).
    pub fn zeros_like(&self) -> Gpt2 {
        let mut z = self.clone();
        z.params_mut().into_iter().for_each(|p| p.fill(0.0));
        z
    }

    /// Total parameter count.
    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.len()).sum()
    }

    /// GPT-style initialisation for training from scratch: weights and embeddings ~ N(0, 0.02)
    /// (approximated by a scaled uniform with the same standard deviation), LayerNorm gains 1,
    /// every bias and shift 0.
    pub fn init(cfg: Config, seed: u64) -> Gpt2 {
        let mut m = Gpt2::random(cfg, seed);
        // uniform on [-1, 1) has std 1/sqrt(3); scale to std 0.02.
        let scale = 0.02 * 3f32.sqrt();
        let layers = cfg.layers;
        for i in 0..m.params().len() {
            let kind = match i {
                0 | 1 => 'w',
                _ if i < 2 + layers * 16 => match (i - 2) % 16 {
                    0 | 10 => 'g',
                    2 | 4 | 6 | 8 | 12 | 14 => 'w',
                    _ => 'b',
                },
                _ if i == 2 + layers * 16 => 'g',
                _ if i == 3 + layers * 16 => 'b',
                _ => 'w',
            };
            let p = &mut m.params_mut()[i];
            match kind {
                'w' => p.iter_mut().for_each(|v| *v *= scale),
                'g' => p.fill(1.0),
                _ => p.fill(0.0),
            }
        }
        m
    }
}
