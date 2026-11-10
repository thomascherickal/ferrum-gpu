//! Parity tolerances, frozen in one place. A later failure is a bug, not a reason to loosen.

/// Relative tolerance for exact-arithmetic ops (add, bias).
pub const EXACT: f32 = 1e-6;
/// Relative tolerance for ops using transcendentals or reductions (gelu, layernorm, softmax...).
/// Calibrated 2026-11-10: worst observed 3.3e-7 on MX330, Intel Xe and llvmpipe; ~30x margin.
pub const REDUCING: f32 = 1e-5;
/// Relative tolerance for whole-model gradients (calibrated 2026-11-10: worst 5.7e-7 on MX330, Xe and llvmpipe).
pub const BACKWARD: f32 = 1e-5;
/// Relative tolerance for a whole forward pass (errors accumulate across layers).
pub const FORWARD: f32 = 1e-5;

/// Largest `|got - want| / max(|want|, 1)`; infinity if the lengths differ or a value is NaN.
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() {
        return f32::INFINITY;
    }
    got.iter().zip(want).fold(0.0, |m, (g, w)| {
        let e = (g - w).abs() / w.abs().max(1.0);
        if e.is_nan() {
            f32::INFINITY
        } else {
            m.max(e)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_rel_err_basics() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!((max_rel_err(&[1.5], &[1.0]) - 0.5).abs() < 1e-7);
        assert!((max_rel_err(&[101.0], &[100.0]) - 0.01).abs() < 1e-6); // relative above 1
        assert_eq!(max_rel_err(&[1.0], &[1.0, 2.0]), f32::INFINITY);
        assert_eq!(max_rel_err(&[f32::NAN], &[1.0]), f32::INFINITY);
        assert_eq!(max_rel_err(&[], &[]), 0.0);
    }
}
