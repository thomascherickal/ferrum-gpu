//! The token embedding table `[vocab, d]` on the device: one buffer, or row blocks when the table
//! is larger than one storage binding (large vocabularies on adapters with small limits).

use super::shard::blocks;
use crate::device::{Gpu, GpuError};
use crate::ops_llama;
use crate::tensor::GpuBuf;

pub(crate) enum Embedding {
    Whole(GpuBuf),
    /// `(first row, rows, block)`.
    Rows(Vec<(usize, usize, GpuBuf)>),
}

impl Embedding {
    /// Upload `tok` (`vocab` rows of `d`), in blocks of at most `limit` bytes.
    pub(super) fn upload(gpu: &Gpu, tok: &[f32], d: usize, limit: u64) -> Result<Self, GpuError> {
        let vocab = tok.len().checked_div(d).unwrap_or(0);
        let rows = blocks(vocab, d as u64 * 4, limit)?;
        if rows.len() <= 1 {
            return Ok(Embedding::Whole(gpu.upload(tok)?));
        }
        let parts = rows
            .into_iter()
            .map(|(r0, n)| Ok((r0, n, gpu.upload(&tok[r0 * d..(r0 + n) * d])?)))
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Embedding::Rows(parts))
    }

    /// `x[i] = table[ids[i]]`. `gi` holds the same ids on the device (used when the table is one
    /// buffer); `d` is the row width.
    pub(super) fn gather(
        &self,
        gpu: &Gpu,
        (ids, gi): (&[usize], &GpuBuf),
        x: &GpuBuf,
        d: usize,
    ) -> Result<(), GpuError> {
        match self {
            Embedding::Whole(t) => ops_llama::gather_rows(gpu, t, gi, x, (ids.len(), d)),
            Embedding::Rows(parts) => {
                for (i, &id) in ids.iter().enumerate() {
                    let (r0, n, block) = parts
                        .iter()
                        .find(|(r0, n, _)| id >= *r0 && id < r0 + n)
                        .ok_or_else(|| {
                        GpuError(format!("token {id} is outside the embedding"))
                    })?;
                    let local = gpu.upload_indices(&[id - r0], *n)?;
                    let row = gpu.zeros(d)?;
                    ops_llama::gather_rows(gpu, block, &local, &row, (1, d))?;
                    gpu.copy_f32(&row, 0, x, i * d, d)?;
                }
                Ok(())
            }
        }
    }

    /// Number of blocks (1 when whole).
    #[cfg(test)]
    pub(super) fn parts(&self) -> usize {
        match self {
            Embedding::Whole(_) => 1,
            Embedding::Rows(p) => p.len(),
        }
    }
}
