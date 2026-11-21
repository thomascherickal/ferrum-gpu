use super::*;

/// The kernel to use for `rows` rows of a `k` x `n` weight, from measurements on two adapters
/// (an Intel Xe and an NVIDIA MX330, `cargo test --release -p ferrum_wgpu --lib -- --ignored
/// bench_quant`, recorded in `docs/gui-studio-plan.md`): the tiled kernel beat the naive one for
/// Q8_0 and F16 on both and was slower or level for Q4_0; the per-column kernel beat it when the
/// output is no wider than the input and lost on the MX330 when it is wider. MXFP4, FP8 and Q2_K
/// (measured 2026-11-18 with the same bench) follow the Q8_0 pattern: tiled faster for several rows
/// by 24% to 87%; per-column faster when `n <= k` by 3% to 78%, and on the MX330 up to 2.5x slower
/// when the output is wider, except Q2_K, which is never slower by more than 1%.
// ponytail: two adapters and two shapes of rule; re-measure on a new adapter with the bench above
// before trusting it there (an Apple GPU has not been run).
pub fn choose(rows: usize, kind: QKind, k: usize, n: usize) -> QPath {
    match (rows, kind) {
        // Q2_K: the per-column kernel was never more than 1% slower than the naive one on either
        // adapter and up to 69% faster, so it takes every single-row shape
        (1, QKind::Q2K) => QPath::Gemv,
        (1, _) if n <= k => QPath::Gemv,
        (1, _) | (_, QKind::Q4_0) => QPath::Naive,
        _ => QPath::Tiled,
    }
}

/// `y[rows, n] = x[rows, k] * W`, dequantising `W` in the inner loop (no bias).
pub fn linear_q(
    gpu: &Gpu,
    x: &GpuBuf,
    w: &QWeight,
    y: &GpuBuf,
    rows: usize,
) -> Result<(), GpuError> {
    linear_q_with(gpu, x, w, y, rows, QPath::Auto)
}

/// [`linear_q`] on a chosen kernel (`Gemv` takes exactly one row).
pub fn linear_q_with(
    gpu: &Gpu,
    x: &GpuBuf,
    w: &QWeight,
    y: &GpuBuf,
    rows: usize,
    path: QPath,
) -> Result<(), GpuError> {
    need("linear_q x", x, rows * w.k)?;
    need("linear_q y", y, rows * w.n)?;
    let uniform = [rows as u32, w.k as u32, w.n as u32, w.kind.id()];
    let buffers = [x, &w.buf, y];
    let path = if path == QPath::Auto {
        choose(rows, w.kind, w.k, w.n)
    } else {
        path
    };
    match path {
        QPath::Naive | QPath::Auto => {
            // the formats added after Q8_0, Q4_0 and F16 keep their one-invocation-per-output kernel apart
            let p = if w.kind.is_core() {
                pipe!(gpu, "quant/qlinear")?
            } else {
                pipe!(gpu, "quant/qlinear_ext")?
            };
            gpu.dispatch_1d(&p, &buffers, &uniform, rows * w.n)
        }
        QPath::Gemv => {
            if rows != 1 {
                return Err(err(format!(
                    "the per-column kernel takes one row, not {rows}"
                )));
            }
            let p = pipe!(gpu, "quant/qgemv")?;
            let x_groups = w.n.clamp(1, 65535) as u32;
            let y_groups = w.n.div_ceil(65535).max(1) as u32;
            gpu.dispatch(&p, &buffers, &uniform, (x_groups, y_groups))
        }
        QPath::Tiled => {
            let p = pipe!(gpu, "quant/qlinear_tiled")?;
            let groups = crate::runtime::groups_for((w.n, rows), (16, 16), gpu.max_workgroups())?;
            gpu.dispatch(&p, &buffers, &uniform, groups)
        }
    }
}
