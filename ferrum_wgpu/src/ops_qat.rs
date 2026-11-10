//! Quantisation-aware training kernel: snap a tensor onto its Q8_0 values on the device.

use crate::device::{Gpu, GpuError};
use crate::ops::one_d;
use crate::tensor::GpuBuf;
use ferrum_ref::qat::eligible;

fn check(src: &GpuBuf, dst: &GpuBuf) -> Result<(), GpuError> {
    if src.len() == dst.len() {
        Ok(())
    } else {
        Err(GpuError(format!(
            "q8_snap: source holds {} elements but destination {}",
            src.len(),
            dst.len()
        )))
    }
}

/// Snap `src` into `dst` along rows: blocks are 32 consecutive elements and `ne0` (the row
/// length) must be a multiple of 32. An ineligible `ne0` copies the data unchanged.
pub fn q8_snap_rows(gpu: &Gpu, src: &GpuBuf, dst: &GpuBuf, ne0: usize) -> Result<(), GpuError> {
    check(src, dst)?;
    if !eligible(ne0) || !src.len().is_multiple_of(32) {
        return gpu.copy_f32(src, 0, dst, 0, src.len());
    }
    let p = gpu.pipeline("q8_snap", include_str!("shaders/q8_snap.wgsl"))?;
    let blocks = src.len() / 32;
    gpu.dispatch(&p, &[src, dst], &[blocks as u32, 0, 0], one_d(gpu, blocks)?)
}

/// Snap an `[rows_in, cols_out]` row-major weight down its columns (GGUF's layout). An
/// ineligible `rows_in` (not a multiple of 32) copies the data unchanged.
pub fn q8_snap_cols(
    gpu: &Gpu,
    src: &GpuBuf,
    dst: &GpuBuf,
    (rows_in, cols_out): (usize, usize),
) -> Result<(), GpuError> {
    check(src, dst)?;
    if src.len() != rows_in * cols_out {
        return Err(GpuError(format!(
            "q8_snap: {} elements is not {rows_in} x {cols_out}",
            src.len()
        )));
    }
    if !eligible(rows_in) {
        return gpu.copy_f32(src, 0, dst, 0, src.len());
    }
    let p = gpu.pipeline("q8_snap", include_str!("shaders/q8_snap.wgsl"))?;
    let blocks = rows_in / 32 * cols_out;
    gpu.dispatch(
        &p,
        &[src, dst],
        &[blocks as u32, 1, cols_out as u32],
        one_d(gpu, blocks)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_for_test;
    use ferrum_ref::matmul::rand_vec;
    use ferrum_ref::qat::{snap_cols, snap_rows};

    /// Values that stress the quantiser: spread magnitudes, ties, zeros, a block of denormal-scale
    /// values (the f16 scale is subnormal there), and large values.
    fn nasty(n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = rand_vec(n, 9).into_iter().map(|x| x * 3.0).collect();
        for (i, x) in v.iter_mut().enumerate() {
            match i / 32 % 7 {
                0 => *x *= 1e-6,                // tiny: the f16 scale becomes subnormal
                1 => *x = 0.0,                  // all-zero block
                2 => *x *= 1e4,                 // large
                3 => *x = (i % 9) as f32 - 4.0, // exact small integers: many ties at .5 steps
                _ => {}
            }
        }
        v
    }

    fn mismatches(a: &[f32], b: &[f32]) -> usize {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    }

    #[test]
    fn values_that_round_to_zero_are_positive_zero_like_the_cpu() {
        // Regression: the first kernel negated a float, so a small negative value came out as
        // -0.0 where the CPU (which quantises to an i8) gives +0.0. Equal as numbers, but the bit
        // patterns differ, which would break exact comparisons and checksums.
        let Some(g) = open_for_test() else { return };
        let mut blk = vec![0.0f32; 32];
        blk[0] = 127.0;
        blk[1] = -0.3; // rounds to q = 0
        blk[2] = -0.0;
        let (src, dst) = (g.upload(&blk).unwrap(), g.zeros(32).unwrap());
        q8_snap_rows(&g, &src, &dst, 32).unwrap();
        let got = g.read(&dst).unwrap();
        assert_eq!(got[1].to_bits(), 0.0f32.to_bits(), "{:e}", got[1]);
        assert_eq!(got[2].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn row_blocks_match_the_cpu_snap_bit_for_bit() {
        let Some(g) = open_for_test() else { return };
        for n in [32usize, 96, 4096, 64 * 100] {
            let data = nasty(n);
            let mut want = data.clone();
            snap_rows(&mut want, 32);
            let (src, dst) = (g.upload(&data).unwrap(), g.zeros(n).unwrap());
            q8_snap_rows(&g, &src, &dst, 32).unwrap();
            let got = g.read(&dst).unwrap();
            let bad = mismatches(&got, &want);
            eprintln!("rows n={n}: {bad} of {n} elements differ from the CPU snap");
            assert_eq!(bad, 0, "n={n}");
        }
    }

    #[test]
    fn column_blocks_match_the_cpu_snap_bit_for_bit() {
        let Some(g) = open_for_test() else { return };
        for (rows_in, cols_out) in [(32usize, 1usize), (32, 7), (64, 33), (96, 128), (256, 64)] {
            let data = nasty(rows_in * cols_out);
            let mut want = data.clone();
            snap_cols(&mut want, rows_in, cols_out);
            let (src, dst) = (g.upload(&data).unwrap(), g.zeros(data.len()).unwrap());
            q8_snap_cols(&g, &src, &dst, (rows_in, cols_out)).unwrap();
            let bad = mismatches(&g.read(&dst).unwrap(), &want);
            eprintln!("cols {rows_in}x{cols_out}: {bad} elements differ");
            assert_eq!(bad, 0, "{rows_in}x{cols_out}");
        }
    }

    #[test]
    fn ties_round_away_from_zero_and_the_scale_is_rounded_to_f16() {
        let Some(g) = open_for_test() else { return };
        // amax 127 -> d = 1 exactly; 2.5 -> 3, -2.5 -> -3 (half away from zero), 0.5 -> 1.
        let mut blk = vec![0.0f32; 32];
        blk[0] = 127.0;
        blk[1] = 2.5;
        blk[2] = -2.5;
        blk[3] = 0.5;
        blk[4] = -0.49999997; // just below the tie: must round to 0, not 1
        let (src, dst) = (g.upload(&blk).unwrap(), g.zeros(32).unwrap());
        q8_snap_rows(&g, &src, &dst, 32).unwrap();
        let got = g.read(&dst).unwrap();
        assert_eq!(&got[..5], &[127.0, 3.0, -3.0, 1.0, 0.0]);
        // A scale that f16 cannot hold exactly: amax/127 = 1/3 -> the snapped value uses f16(1/3).
        let mut blk = vec![0.0f32; 32];
        blk[0] = 127.0 / 3.0;
        blk[1] = 1.0;
        let (src, dst) = (g.upload(&blk).unwrap(), g.zeros(32).unwrap());
        q8_snap_rows(&g, &src, &dst, 32).unwrap();
        let mut want = blk.clone();
        snap_rows(&mut want, 32);
        assert_eq!(g.read(&dst).unwrap(), want);
    }

    #[test]
    fn the_snap_is_idempotent_on_the_device() {
        let Some(g) = open_for_test() else { return };
        let data = rand_vec(64 * 4, 12);
        let (a, b, c) = (
            g.upload(&data).unwrap(),
            g.zeros(256).unwrap(),
            g.zeros(256).unwrap(),
        );
        q8_snap_cols(&g, &a, &b, (64, 4)).unwrap();
        q8_snap_cols(&g, &b, &c, (64, 4)).unwrap();
        assert_eq!(g.read(&b).unwrap(), g.read(&c).unwrap());
    }

    #[test]
    fn ineligible_shapes_are_copied_unchanged() {
        let Some(g) = open_for_test() else { return };
        let data = rand_vec(8 * 5, 13);
        let (src, dst) = (g.upload(&data).unwrap(), g.zeros(40).unwrap());
        q8_snap_cols(&g, &src, &dst, (8, 5)).unwrap();
        assert_eq!(g.read(&dst).unwrap(), data);
        let (src, dst) = (g.upload(&data).unwrap(), g.zeros(40).unwrap());
        q8_snap_rows(&g, &src, &dst, 8).unwrap();
        assert_eq!(g.read(&dst).unwrap(), data);
        // Eligible row length but a tensor that is not a whole number of blocks.
        let odd = rand_vec(40, 14);
        let (src, dst) = (g.upload(&odd).unwrap(), g.zeros(40).unwrap());
        q8_snap_rows(&g, &src, &dst, 32).unwrap();
        assert_eq!(g.read(&dst).unwrap(), odd);
    }

    #[test]
    fn shape_errors_and_faults_are_reported() {
        let Some(g) = open_for_test() else { return };
        let (a, b) = (g.zeros(64).unwrap(), g.zeros(32).unwrap());
        assert!(q8_snap_rows(&g, &a, &b, 32)
            .unwrap_err()
            .to_string()
            .contains("destination"));
        assert!(q8_snap_cols(&g, &a, &b, (32, 2)).is_err());
        let c = g.zeros(64).unwrap();
        assert!(q8_snap_cols(&g, &a, &c, (32, 3))
            .unwrap_err()
            .to_string()
            .contains("is not 32 x 3"));
        assert_eq!(g.sweep_faults(|| q8_snap_cols(&g, &a, &c, (32, 2))), 1);
        assert_eq!(g.sweep_faults(|| q8_snap_rows(&g, &a, &c, 32)), 1);
        assert_eq!(
            g.sweep_faults(|| q8_snap_rows(&g, &a, &c, 8)),
            1,
            "the copy path is a command too"
        );
    }
}
