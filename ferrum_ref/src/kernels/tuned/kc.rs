use super::*;

pub(super) const KC: usize = 256;
pub(super) const NC: usize = 256;
pub(super) const MR: usize = 4;
pub(super) const NR: usize = 8;

/// Accumulate rows `r0..r1` of `A*B` into `out` (those rows, indexed locally, pre-zeroed).
pub(super) fn block(
    a: &[f32],
    b: &[f32],
    (k, n): (usize, usize),
    (r0, r1): (usize, usize),
    out: &mut [f32],
) {
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
pub(super) fn micro(
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
