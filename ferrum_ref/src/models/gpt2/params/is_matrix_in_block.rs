use super::*;

impl Block {
    /// The 16 parameters in canonical order.
    pub fn params(&self) -> [&Vec<f32>; 16] {
        [
            &self.ln1_g,
            &self.ln1_b,
            &self.wq,
            &self.bq,
            &self.wk,
            &self.bk,
            &self.wv,
            &self.bv,
            &self.wo,
            &self.bo,
            &self.ln2_g,
            &self.ln2_b,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
        ]
    }

    /// Mutable access to the 16 parameters in canonical order.
    pub fn params_mut(&mut self) -> [&mut Vec<f32>; 16] {
        [
            &mut self.ln1_g,
            &mut self.ln1_b,
            &mut self.wq,
            &mut self.bq,
            &mut self.wk,
            &mut self.bk,
            &mut self.wv,
            &mut self.bv,
            &mut self.wo,
            &mut self.bo,
            &mut self.ln2_g,
            &mut self.ln2_b,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
        ]
    }
}

/// True for tensors AdamW decays: weight matrices, never biases or LayerNorm parameters
/// (the canonical block order puts matrices at indices 2, 4, 6, 8, 12, 14).
pub fn is_matrix_in_block(i: usize) -> bool {
    matches!(i, 2 | 4 | 6 | 8 | 12 | 14)
}

/// Element count of every parameter in canonical order, or `None` if the config is invalid or
/// a size overflows (so a hostile config can never drive a huge allocation).
pub fn param_lens(c: &Config) -> Option<Vec<usize>> {
    c.validate().ok()?;
    let (d, h) = (c.dim, c.hidden);
    let mul = |a: usize, b: usize| a.checked_mul(b);
    let mut v = vec![mul(c.vocab, d)?, mul(c.ctx, d)?];
    for _ in 0..c.layers {
        let (dd, dh) = (mul(d, d)?, mul(d, h)?);
        v.extend([d, d, dd, d, dd, d, dd, d, dd, d, d, d, dh, h, dh, d]);
    }
    v.extend([d, d, mul(d, c.vocab)?]);
    Some(v)
}
