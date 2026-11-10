// Provenance: logic copied from ferrum-cpu `ferrum_core/src/ops.rs` (`matmul`,
// i-k-j loop order), commit 56a9d3a. Rewritten over plain slices; ours from here on.
//! Row-major f32 matrix multiply `[m,k] x [k,n] -> [m,n]`.

/// Shape error for [`matmul`] and [`matmul_threaded`].
#[derive(Debug, PartialEq, Eq)]
pub enum MatmulError {
    /// A slice length does not match its declared shape.
    Len {
        /// Which operand (`"a"` or `"b"`).
        which: &'static str,
        /// Expected element count.
        want: usize,
        /// Actual element count.
        got: usize,
    },
}

impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let MatmulError::Len { which, want, got } = self;
        write!(
            f,
            "matmul: operand {which} has {got} elements, shape needs {want}"
        )
    }
}

impl std::error::Error for MatmulError {}

fn check(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<(), MatmulError> {
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
    Ok(())
}

fn rows(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32]) {
    for (i, row) in out.chunks_mut(n.max(1)).enumerate() {
        for p in 0..k {
            let av = a[i * k + p];
            let brow = &b[p * n..(p + 1) * n];
            for (o, &bv) in row.iter_mut().zip(brow) {
                *o += av * bv;
            }
        }
    }
}

/// Single-threaded reference matmul. Accumulates in ascending `k` for every element.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, MatmulError> {
    check(a, b, m, k, n)?;
    let mut out = vec![0.0; m * n];
    if n > 0 {
        rows(a, b, k, n, &mut out);
    }
    Ok(out)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        assert_eq!(
            matmul(&a, &b, 2, 3, 2).unwrap(),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn non_square_and_one_row() {
        let a = rand_vec(7, 1);
        let b = rand_vec(7 * 5, 2);
        let got = matmul(&a, &b, 1, 7, 5).unwrap();
        for j in 0..5 {
            let want: f32 = (0..7).map(|p| a[p] * b[p * 5 + j]).sum();
            assert!((got[j] - want).abs() < 1e-5);
        }
    }

    #[test]
    fn shape_errors_name_the_operand() {
        assert_eq!(
            matmul(&[1.0], &[1.0, 2.0], 2, 1, 1),
            Err(MatmulError::Len {
                which: "a",
                want: 2,
                got: 1
            })
        );
        let e = matmul(&[1.0], &[1.0, 2.0], 1, 1, 1).unwrap_err();
        assert!(e.to_string().contains("operand b"));
    }

    #[test]
    fn empty_dimensions_give_empty_or_zero_output() {
        assert_eq!(matmul(&[], &[], 0, 3, 0).unwrap(), Vec::<f32>::new());
        assert_eq!(matmul(&[], &[], 2, 0, 2).unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn threaded_is_bit_identical_for_any_thread_count() {
        let (m, k, n) = (13, 9, 11);
        let a = rand_vec(m * k, 3);
        let b = rand_vec(k * n, 4);
        let want = matmul(&a, &b, m, k, n).unwrap();
        for t in [0, 1, 2, 5, 64] {
            assert_eq!(
                matmul_threaded(&a, &b, m, k, n, t).unwrap(),
                want,
                "threads {t}"
            );
        }
        assert!(matmul_threaded(&a, &b, m, k + 1, n, 2).is_err());
    }

    #[test]
    fn rand_vec_is_deterministic_and_in_range() {
        assert_eq!(rand_vec(8, 7), rand_vec(8, 7));
        assert_ne!(rand_vec(8, 7), rand_vec(8, 8));
        assert!(rand_vec(1000, 9).iter().all(|v| (-1.0..1.0).contains(v)));
    }
}
