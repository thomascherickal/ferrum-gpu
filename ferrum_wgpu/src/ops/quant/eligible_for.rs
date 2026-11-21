use super::*;

/// Whether a matrix with input width `k` fills whole blocks of `kind`.
pub fn eligible_for(kind: QKind, k: usize) -> bool {
    k > 0 && k.is_multiple_of(kind.elems())
}

/// The format a matrix of input width `k` is actually stored in when `kind` is requested: Q2_K
/// needs 256-wide blocks, so a width that is a whole number of 32s uses Q4_0 instead (llama.cpp's own
/// fallback, and what the file writer does); everything else is `kind`.
pub fn effective(kind: QKind, k: usize) -> QKind {
    if kind == QKind::Q2K && !eligible_for(kind, k) && eligible(k) {
        QKind::Q4_0
    } else {
        kind
    }
}

/// Device words a `k x n` weight occupies, or `None` if `k` is not a whole number of blocks or
/// the size overflows.
pub fn packed_words(kind: QKind, k: usize, n: usize) -> Option<usize> {
    if !eligible_for(kind, k) {
        return None;
    }
    (k / kind.elems()).checked_mul(n)?.checked_mul(kind.words())
}

/// A quantised weight resident on the device: `n` output columns of `k` inputs each.
pub struct QWeight {
    pub(crate) buf: GpuBuf,
    pub(crate) kind: QKind,
    pub(crate) k: usize,
    pub(crate) n: usize,
}
