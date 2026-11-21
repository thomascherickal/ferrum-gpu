use super::*;

/// What the GPU actually computes when `quant` is requested: `m` with every matrix that
/// can be quantised (input width a multiple of 32) replaced by its dequantised value. A model
/// loaded from a Q8_0/Q4_0 GGUF is already in this form (requantising it is the identity), so
/// this is the oracle for a quantised run of an f32 model.
pub fn dequantised(m: &Llama, quant: QKind) -> Result<Llama, GpuError> {
    let snap = |w: &[f32], k: usize, n: usize| -> Result<Vec<f32>, GpuError> {
        let quant = effective(quant, k);
        if !eligible_for(quant, k) {
            return Ok(w.to_vec());
        }
        let bytes = quantise_matrix(quant, w, k, n)?;
        let rows = dequantise_bytes(quant, &bytes, k * n)?;
        Ok(transpose(&rows, n, k))
    };
    let c = &m.cfg;
    let mut out = m.clone();
    for b in &mut out.blocks {
        b.wq = snap(&b.wq, c.dim, c.q_dim())?;
        b.wk = snap(&b.wk, c.dim, c.kv_dim())?;
        b.wv = snap(&b.wv, c.dim, c.kv_dim())?;
        b.wo = snap(&b.wo, c.q_dim(), c.dim)?;
        b.gate = snap(&b.gate, c.dim, c.hidden)?;
        b.up = snap(&b.up, c.dim, c.hidden)?;
        b.down = snap(&b.down, c.hidden, c.dim)?;
    }
    let head = snap(&m.head, c.dim, c.vocab)?;
    out.tied = out.tied && head == m.head;
    out.head = head;
    Ok(out)
}
