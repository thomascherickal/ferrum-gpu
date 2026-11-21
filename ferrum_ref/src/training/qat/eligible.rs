use super::*;

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
