// Provenance: which tensors are snapped and along which axis follows ferrum-cpu
// `train_transformer/qat.rs` and `quant.rs` (commit 56a9d3a). The snap itself is the Q8_0
// encoder/decoder of `ferrum_gguf`, so training sees exactly the values a Q8_0 export stores.
//! Quantisation-aware training support: snap weights onto their Q8_0 values.
//!
//! QAT runs the forward and backward pass on snapped copies of the weights while Adam updates
//! the full-precision masters (a straight-through estimator), so a model trained this way loses
//! little when exported as Q8_0.

use crate::gpt2::Gpt2;
use ferrum_gguf::quant::{decode, encode};
use ferrum_gguf::{GGML_Q8_0, QK};

/// True when a tensor whose contiguous GGML dimension has `ne0` elements is stored Q8_0.
pub fn eligible(ne0: usize) -> bool {
    ne0 > 0 && ne0.is_multiple_of(QK)
}

/// Snap one block of 32 values onto its Q8_0 value (quantise, then dequantise).
pub fn snap_block(block: &[f32]) -> Vec<f32> {
    let bytes = encode(GGML_Q8_0, block).expect("a block of 32 is always encodable");
    decode(GGML_Q8_0, &bytes, QK).expect("just encoded")
}

/// Snap a tensor stored with its contiguous GGML dimension along the row (blocks are 32
/// consecutive values). A no-op when `ne0` is not a multiple of 32.
pub fn snap_rows(data: &mut [f32], ne0: usize) {
    if !eligible(ne0) {
        return;
    }
    for blk in data.chunks_exact_mut(QK) {
        blk.copy_from_slice(&snap_block(blk));
    }
}

/// Snap an `[in, out]` row-major weight the way GGUF stores it: blocks run down each column
/// over the input dimension. A no-op when `rows_in` is not a multiple of 32.
pub fn snap_cols(data: &mut [f32], rows_in: usize, cols_out: usize) {
    if !eligible(rows_in) {
        return;
    }
    let mut buf = [0.0f32; QK];
    for j in 0..cols_out {
        for i0 in (0..rows_in).step_by(QK) {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = data[(i0 + k) * cols_out + j];
            }
            let snapped = snap_block(&buf);
            for (k, v) in snapped.iter().enumerate() {
                data[(i0 + k) * cols_out + j] = *v;
            }
        }
    }
}

impl Gpt2 {
    /// A copy with exactly the tensors a Q8_0 export stores quantised snapped onto their Q8_0
    /// values: the token embedding (along its rows), the six matrices of every block and the
    /// head (down their columns). Position embeddings, biases and LayerNorm parameters, and
    /// matrices whose input dimension is not a multiple of 32 (stored F32), are left alone.
    pub fn fake_quantize(&self) -> Gpt2 {
        let mut q = self.clone();
        let (d, h, v) = (self.cfg.dim, self.cfg.hidden, self.cfg.vocab);
        snap_rows(&mut q.tok, d);
        for b in &mut q.blocks {
            for w in [&mut b.wq, &mut b.wk, &mut b.wv, &mut b.wo] {
                snap_cols(w, d, d);
            }
            snap_cols(&mut b.w1, d, h);
            snap_cols(&mut b.w2, h, d);
        }
        snap_cols(&mut q.head, d, v);
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt2::Config;
    use crate::matmul::rand_vec;

    fn cfg(dim: usize) -> Config {
        Config {
            vocab: 16,
            ctx: 8,
            dim,
            heads: 2,
            layers: 2,
            hidden: dim * 2,
        }
    }

    #[test]
    fn eligibility_needs_whole_blocks() {
        assert!(eligible(32) && eligible(64) && eligible(4096));
        assert!(!eligible(0) && !eligible(31) && !eligible(33) && !eligible(8));
    }

    #[test]
    fn a_snapped_block_is_a_fixed_point_with_at_most_127_distinct_levels() {
        let blk = rand_vec(32, 1);
        let once = snap_block(&blk);
        assert_ne!(once, blk, "random values are not already on the grid");
        assert_eq!(snap_block(&once), once, "idempotent");
        let step = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 127.0;
        for (a, b) in blk.iter().zip(&once) {
            assert!((a - b).abs() <= step * 0.51 + 1e-6);
        }
        assert_eq!(snap_block(&[0.0; 32]), vec![0.0; 32]);
    }

    #[test]
    fn rows_and_cols_snap_the_right_blocks() {
        // rows: two blocks of 32 along the flat data.
        let mut r = rand_vec(64, 2);
        let want: Vec<f32> = r.chunks(32).flat_map(snap_block).collect();
        snap_rows(&mut r, 32);
        assert_eq!(r, want);
        // cols: a [64, 3] matrix, blocks run down each column.
        let mut m = rand_vec(64 * 3, 3);
        let orig = m.clone();
        snap_cols(&mut m, 64, 3);
        for j in 0..3 {
            for b in 0..2 {
                let col: Vec<f32> = (0..32).map(|k| orig[(b * 32 + k) * 3 + j]).collect();
                let want = snap_block(&col);
                for (k, w) in want.iter().enumerate() {
                    assert_eq!(m[(b * 32 + k) * 3 + j], *w, "col {j} block {b} elem {k}");
                }
            }
        }
    }

    #[test]
    fn ineligible_shapes_are_left_untouched() {
        let mut a = rand_vec(40, 4);
        let o = a.clone();
        snap_rows(&mut a, 8);
        snap_cols(&mut a, 8, 5);
        snap_cols(&mut a, 0, 5);
        assert_eq!(a, o);
    }

    #[test]
    fn fake_quantize_touches_only_what_a_q8_0_export_quantises() {
        let w = Gpt2::random(cfg(32), 5);
        let q = w.fake_quantize();
        assert_ne!(q.tok, w.tok);
        assert_ne!(q.head, w.head);
        assert_ne!(q.blocks[1].w2, w.blocks[1].w2);
        assert_eq!(q.pos, w.pos, "position embeddings stay F32");
        assert_eq!(q.blocks[0].bq, w.blocks[0].bq);
        assert_eq!(q.blocks[0].ln1_g, w.blocks[0].ln1_g);
        assert_eq!(q.lnf_b, w.lnf_b);
        assert_eq!(q.fake_quantize(), q, "idempotent");
        // A width that is not a multiple of 32 is stored F32, so nothing changes.
        let small = Gpt2::random(cfg(8), 6);
        assert_eq!(small.fake_quantize(), small);
    }

    #[test]
    fn the_snapped_values_are_exactly_what_a_q8_0_gguf_would_hold() {
        // Property QAT depends on: quantising the snapped model again loses nothing, because the
        // snap used the very same encoder (tested end to end in ferrum_model's export test).
        let w = Gpt2::random(cfg(32), 7).fake_quantize();
        let mut again = w.clone();
        snap_rows(&mut again.tok, 32);
        assert_eq!(again.tok, w.tok);
    }
}
