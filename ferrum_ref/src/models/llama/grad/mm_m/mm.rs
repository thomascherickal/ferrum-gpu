use super::*;

pub(crate) fn mm(
    a: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
    threads: usize,
) -> Vec<f32> {
    matmul_tuned(a, b, m, k, n, threads)
        .unwrap_or_else(|e: MatmulError| panic!("internal shape error: {e}"))
}

pub(crate) fn lin(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    dims: (usize, usize, usize),
    threads: usize,
) -> Vec<f32> {
    let y = mm(x, w, dims, threads);
    if b.is_empty() {
        y
    } else {
        nn::add_bias(&y, b)
    }
}

/// RMSNorm backward: returns `(dx, dgain)`. With `inv = (mean(x^2) + eps)^-1/2`,
/// `dx = inv * (dy*g) - x * inv^3 / n * sum(dy*g*x)` per row.
pub fn rmsnorm_bwd(
    x: &[f32],
    g: &[f32],
    dy: &[f32],
    (rows, cols): (usize, usize),
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (mut dx, mut dg) = (vec![0.0; rows * cols], vec![0.0; cols]);
    for r in 0..rows {
        let (xr, dyr) = (&x[r * cols..(r + 1) * cols], &dy[r * cols..(r + 1) * cols]);
        let ms = xr.iter().map(|&v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let dot: f32 = (0..cols).map(|c| dyr[c] * g[c] * xr[c]).sum();
        let k = inv * inv * inv * dot / cols as f32;
        for c in 0..cols {
            dx[r * cols + c] = inv * dyr[c] * g[c] - xr[c] * k;
            dg[c] += dyr[c] * xr[c] * inv;
        }
    }
    (dx, dg)
}

/// SwiGLU backward: returns `(dgate, dup)` for `act = silu(gate) * up`.
pub fn swiglu_bwd(gate: &[f32], up: &[f32], dact: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut dgate = Vec::with_capacity(gate.len());
    let mut dup = Vec::with_capacity(gate.len());
    for ((&g, &u), &d) in gate.iter().zip(up).zip(dact) {
        let s = 1.0 / (1.0 + (-g).exp());
        dgate.push(d * u * s * (1.0 + g * (1.0 - s)));
        dup.push(d * silu(g));
    }
    (dgate, dup)
}

/// Gradient through [`apply_rope`]: the transpose of a rotation is the rotation by the opposite
/// angle, so this applies the same tables with the sine negated.
pub fn rope_bwd(
    dy: &mut [f32],
    dims: (usize, usize, usize, usize),
    (cos, sin): (&[f32], &[f32]),
    ty: RopeType,
) {
    let neg: Vec<f32> = sin.iter().map(|s| -s).collect();
    apply_rope(dy, dims, (cos, &neg), ty);
}
