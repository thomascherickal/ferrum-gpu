use super::*;

/// RMSNorm backward. `stats` is scratch of `rows` (the per-row inverse RMS).
pub fn rmsnorm_bwd(
    gpu: &Gpu,
    (x, gain, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dgain): (&GpuBuf, &GpuBuf),
    stats: &GpuBuf,
    (rows, cols): (usize, usize),
    eps: f32,
) -> Result<(), GpuError> {
    for (n, b) in [("x", x), ("dy", dy), ("dx", dx)] {
        need(&format!("rmsnorm_bwd {n}"), b, rows * cols)?;
    }
    need("rmsnorm_bwd gain", gain, cols)?;
    need("rmsnorm_bwd dgain", dgain, cols)?;
    need("rmsnorm_bwd stats", stats, rows)?;
    let p1 = pipe!(gpu, "llama/rms_bwd_dx")?;
    gpu.dispatch_1d(
        &p1,
        &[x, gain, dy, dx, stats],
        &[rows as u32, cols as u32, eps.to_bits()],
        rows,
    )?;
    let p2 = pipe!(gpu, "llama/rms_bwd_gain")?;
    gpu.dispatch_1d(
        &p2,
        &[x, dy, stats, dgain],
        &[rows as u32, cols as u32],
        cols,
    )
}

/// SwiGLU backward: `dgate` and `dup` from `gate`, `up` and `dact`.
pub fn swiglu_bwd(
    gpu: &Gpu,
    (gate, up, dact): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dgate, dup): (&GpuBuf, &GpuBuf),
) -> Result<(), GpuError> {
    for (n, b) in [("up", up), ("dact", dact), ("dgate", dgate), ("dup", dup)] {
        need(&format!("swiglu_bwd {n}"), b, gate.len())?;
    }
    let p = pipe!(gpu, "llama/swiglu_bwd")?;
    gpu.dispatch_1d(
        &p,
        &[gate, up, dact, dgate, dup],
        &[gate.len() as u32],
        gate.len(),
    )
}
