// Provenance: algorithm copied from ferrum-cpu `ferrum_core/src/ops.rs` (matmul_block,
// micro_tile, edge_rows; KC/NC cache panels, 4x8 register tile), commit 56a9d3a. Rewritten
// over plain slices with a scoped-thread row split; ours from here on.
//! Tuned CPU matmul: the speed baseline for GPU gates (the simple kernel in [`crate::matmul`]
//! stays the parity oracle). For any output element the contraction sums `k` in ascending
//! order, so the result is bit-identical to [`crate::matmul::matmul`] at any thread count.

use crate::matmul::MatmulError;

const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 4;
const NR: usize = 8;

/// Accumulate rows `r0..r1` of `A*B` into `out` (those rows, indexed locally, pre-zeroed).
fn block(a: &[f32], b: &[f32], (k, n): (usize, usize), (r0, r1): (usize, usize), out: &mut [f32]) {
    let mut jj = 0;
    while jj < n {
        let j1 = (jj + NC).min(n);
        let mut pp = 0;
        while pp < k {
            let p1 = (pp + KC).min(k);
            let mut i = r0;
            while i + MR <= r1 {
                let mut j = jj;
                while j + NR <= j1 {
                    micro(a, b, (k, n), (i, r0), j, (pp, p1), out);
                    j += NR;
                }
                if j < j1 {
                    edge(a, b, (k, n), (i, i + MR), r0, (j, j1), (pp, p1), out);
                }
                i += MR;
            }
            if i < r1 {
                edge(a, b, (k, n), (i, r1), r0, (jj, j1), (pp, p1), out);
            }
            pp = p1;
        }
        jj = j1;
    }
}

/// `MR x NR` register tile over `p in pp..p1`, in ascending order, onto the values in `out`.
fn micro(
    a: &[f32],
    b: &[f32],
    (k, n): (usize, usize),
    (i, r0): (usize, usize),
    j: usize,
    (pp, p1): (usize, usize),
    out: &mut [f32],
) {
    let mut acc = [[0.0f32; NR]; MR];
    for (r, row) in acc.iter_mut().enumerate() {
        let o = (i + r - r0) * n + j;
        row.copy_from_slice(&out[o..o + NR]);
    }
    let a_rows: [&[f32]; MR] = std::array::from_fn(|r| &a[(i + r) * k + pp..(i + r) * k + p1]);
    for (q, p) in (pp..p1).enumerate() {
        let bs = &b[p * n + j..p * n + j + NR];
        for r in 0..MR {
            let av = a_rows[r][q];
            for c in 0..NR {
                acc[r][c] += av * bs[c];
            }
        }
    }
    for (r, row) in acc.iter().enumerate() {
        let o = (i + r - r0) * n + j;
        out[o..o + NR].copy_from_slice(row);
    }
}

/// Plain accumulation for rows `ia..ib`, columns `ja..jb` that do not fill a register tile.
#[allow(clippy::too_many_arguments)]
fn edge(
    a: &[f32],
    b: &[f32],
    (k, n): (usize, usize),
    (ia, ib): (usize, usize),
    r0: usize,
    (ja, jb): (usize, usize),
    (pp, p1): (usize, usize),
    out: &mut [f32],
) {
    for i in ia..ib {
        for p in pp..p1 {
            let a_ip = a[i * k + p];
            let o = &mut out[(i - r0) * n + ja..(i - r0) * n + jb];
            for (ov, &bv) in o.iter_mut().zip(&b[p * n + ja..p * n + jb]) {
                *ov += a_ip * bv;
            }
        }
    }
}

/// Tuned `[m,k] x [k,n] -> [m,n]` on `threads` scoped threads split by output rows.
pub fn matmul_tuned(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<f32>, MatmulError> {
    if a.len() != m * k {
        return Err(MatmulError::Len {
            which: "a",
            want: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(MatmulError::Len {
            which: "b",
            want: k * n,
            got: b.len(),
        });
    }
    let mut out = vec![0.0; m * n];
    if m == 0 || n == 0 {
        return Ok(out);
    }
    let per = m.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * n).enumerate() {
            let r0 = ci * per;
            let r1 = (r0 + per).min(m);
            s.spawn(move || block(a, b, (k, n), (r0, r1), chunk));
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::{matmul, rand_vec};

    #[test]
    fn bit_identical_to_the_simple_kernel_on_awkward_shapes_and_thread_counts() {
        // Shapes straddle the 4x8 register tile and the 256 cache panels.
        for &(m, k, n) in &[
            (1, 1, 1),
            (3, 5, 7),
            (4, 8, 8),
            (5, 9, 17),
            (9, 300, 11),
            (13, 513, 261),
            (64, 64, 64),
        ] {
            let (a, b) = (rand_vec(m * k, 1), rand_vec(k * n, 2));
            let want = matmul(&a, &b, m, k, n).unwrap();
            for t in [1, 2, 3, 8, 100] {
                assert_eq!(
                    matmul_tuned(&a, &b, m, k, n, t).unwrap(),
                    want,
                    "{m}x{k}x{n} threads {t}"
                );
            }
        }
    }

    #[test]
    fn empty_and_mismatched_inputs() {
        assert!(matmul_tuned(&[], &[], 0, 3, 0, 4).unwrap().is_empty());
        assert_eq!(matmul_tuned(&[], &[], 2, 0, 2, 2).unwrap(), vec![0.0; 4]);
        assert!(matmul_tuned(&[1.0], &[1.0, 2.0], 1, 2, 1, 1).is_err());
        assert!(matmul_tuned(&[1.0, 2.0], &[1.0], 1, 2, 1, 1).is_err());
    }
}
