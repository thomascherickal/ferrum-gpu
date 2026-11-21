use super::*;

/// Plain accumulation for rows `ia..ib`, columns `ja..jb` that do not fill a register tile.
#[allow(clippy::too_many_arguments)]
pub(super) fn edge(
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
