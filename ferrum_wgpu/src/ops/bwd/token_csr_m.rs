use super::*;

/// The CSR index the token-embedding gradient needs: `start[v]..start[v+1]` indexes `order`,
/// which lists the rows holding token `v` in ascending row order (a stable counting sort).
pub fn token_csr(ids: &[usize], vocab: usize) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
    if let Some(bad) = ids.iter().find(|&&i| i >= vocab) {
        return Err(err(format!(
            "token {bad} is out of range for vocab {vocab}"
        )));
    }
    let mut start = vec![0u32; vocab + 1];
    for &i in ids {
        start[i + 1] += 1;
    }
    for v in 0..vocab {
        start[v + 1] += start[v];
    }
    let mut next = start.clone();
    let mut order = vec![0u32; ids.len()];
    for (r, &i) in ids.iter().enumerate() {
        order[next[i] as usize] = r as u32;
        next[i] += 1;
    }
    Ok((start, order))
}

/// Embedding gradients from `dx` (`rows x c`): `dtok [vocab x c]` and `dpos [ctx x c]`.
/// `ids` are the host-side token ids (the CSR index is built from them).
pub fn embed_bwd(
    gpu: &Gpu,
    dx: &GpuBuf,
    ids: &[usize],
    (dtok, dpos): (&GpuBuf, &GpuBuf),
    (vocab, c, t, ctx): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let rows = ids.len();
    need("embed_bwd dx", dx, rows * c)?;
    need("embed_bwd dtok", dtok, vocab * c)?;
    need("embed_bwd dpos", dpos, ctx * c)?;
    if t == 0 || t > ctx {
        return Err(err(format!(
            "embed_bwd: sequence length {t} must be in 1..={ctx}"
        )));
    }
    let (start, order) = token_csr(ids, vocab)?;
    let (gs, go) = (
        gpu.upload_raw(bytemuck::cast_slice(&start), start.len())?,
        gpu.upload_raw(bytemuck::cast_slice(&order), order.len())?,
    );
    let p = pipe!(gpu, "bwd/embed_bwd_tok")?;
    gpu.dispatch_1d(
        &p,
        &[dx, &gs, &go, dtok],
        &[vocab as u32, c as u32],
        vocab * c,
    )?;
    let p = pipe!(gpu, "bwd/embed_bwd_pos")?;
    gpu.dispatch_1d(
        &p,
        &[dx, dpos],
        &[rows as u32, c as u32, t as u32, ctx as u32],
        ctx * c,
    )
}
