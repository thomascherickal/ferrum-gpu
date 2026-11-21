use super::*;

impl Llama {
    /// All trainable parameters in canonical order: token embedding, each block's
    /// [`BLOCK_PARAMS`], output norm, head. (A tied head is not a separate parameter; training
    /// refuses tied models.)
    pub fn params(&self) -> Vec<&Vec<f32>> {
        let mut v = vec![&self.tok];
        self.blocks.iter().for_each(|b| v.extend(b.params()));
        v.extend([&self.out_norm, &self.head]);
        v
    }

    /// Mutable parameters in canonical order.
    pub fn params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v = vec![&mut self.tok];
        self.blocks
            .iter_mut()
            .for_each(|b| v.extend(b.params_mut()));
        v.extend([&mut self.out_norm, &mut self.head]);
        v
    }

    /// A model of the same shape with every parameter zero (a gradient or moment container).
    pub fn zeros_like(&self) -> Llama {
        let mut z = self.clone();
        z.params_mut().into_iter().for_each(|p| p.fill(0.0));
        z
    }

    /// Whether AdamW decays canonical parameter `i`: weight matrices and the embedding are
    /// decayed; norm gains and biases are not.
    pub fn decays(&self, i: usize) -> bool {
        let n = self.blocks.len() * BLOCK_PARAMS;
        match i {
            0 => true,
            _ if i < 1 + n => matches!((i - 1) % BLOCK_PARAMS, 1 | 3 | 5 | 7 | 9 | 10 | 11),
            _ if i == 1 + n => false, // output norm
            _ => true,                // head
        }
    }
}
