use super::*;

/// How the canonical parameter `i` of a model is treated by the snap.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Snap {
    /// Token embedding, blocks of 32 along each row (row length `dim`).
    Rows(usize),
    /// An `[in, out]` matrix, blocks down each column.
    Cols(usize, usize),
    /// Stored F32 in a Q8_0 export (position embeddings, biases, LayerNorm): copied unchanged.
    Keep,
}

/// What the snap does to canonical parameter `i`, mirroring which tensors the GGUF writer stores
/// as Q8_0.
pub(super) fn snap_kind(c: &Config, i: usize) -> Snap {
    let (d, h, v) = (c.dim, c.hidden, c.vocab);
    let blocks_end = 2 + 16 * c.layers;
    match i {
        0 => Snap::Rows(d),
        1 => Snap::Keep,
        _ if i < blocks_end => match (i - 2) % 16 {
            2 | 4 | 6 | 8 => Snap::Cols(d, d),
            12 => Snap::Cols(d, h),
            14 => Snap::Cols(h, d),
            _ => Snap::Keep,
        },
        _ if i == blocks_end + 2 => Snap::Cols(d, v),
        _ => Snap::Keep,
    }
}

/// The snapped copies QAT keeps: one buffer per parameter (unsnapped ones are plain copies, so
/// the forward pass can index one list).
pub fn qat_plan(c: &Config) -> BufferPlan {
    let mut p = BufferPlan::default();
    for (i, len) in ferrum_ref::gpt2::param_lens(c)
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        p.add(&format!("fwd{i}"), (len * 4) as u64);
    }
    p
}
