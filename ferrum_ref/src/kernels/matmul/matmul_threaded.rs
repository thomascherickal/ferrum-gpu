use super::*;

/// Same arithmetic split by output rows over `threads` scoped threads. Bit-identical
/// to [`matmul`] for any thread count (rows are independent).
pub fn matmul_threaded(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<f32>, MatmulError> {
    check(a, b, m, k, n)?;
    let mut out = vec![0.0; m * n];
    if m == 0 || n == 0 {
        return Ok(out);
    }
    let per = m.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * n).enumerate() {
            let a_part = &a[ci * per * k..];
            s.spawn(move || rows(a_part, b, k, n, chunk));
        }
    });
    Ok(out)
}

/// Deterministic pseudo-random values in `[-1, 1)` (xorshift64) for tests and benches.
pub fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
