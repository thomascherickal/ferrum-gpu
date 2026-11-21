use super::*;

/// Parameters per block in canonical order (see [`Llama::params`]).
pub const BLOCK_PARAMS: usize = 12;

impl Block {
    /// The block's parameters in canonical order: attention norm, `wq`, `bq`, `wk`, `bk`, `wv`,
    /// `bv`, `wo`, FFN norm, `gate`, `up`, `down`. An absent bias is an empty vector.
    pub fn params(&self) -> [&Vec<f32>; BLOCK_PARAMS] {
        [
            &self.attn_norm,
            &self.wq,
            &self.bq,
            &self.wk,
            &self.bk,
            &self.wv,
            &self.bv,
            &self.wo,
            &self.ffn_norm,
            &self.gate,
            &self.up,
            &self.down,
        ]
    }

    /// Mutable access in canonical order.
    pub fn params_mut(&mut self) -> [&mut Vec<f32>; BLOCK_PARAMS] {
        [
            &mut self.attn_norm,
            &mut self.wq,
            &mut self.bq,
            &mut self.wk,
            &mut self.bk,
            &mut self.wv,
            &mut self.bv,
            &mut self.wo,
            &mut self.ffn_norm,
            &mut self.gate,
            &mut self.up,
            &mut self.down,
        ]
    }
}
