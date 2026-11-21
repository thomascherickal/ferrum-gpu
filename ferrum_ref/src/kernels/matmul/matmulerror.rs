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

pub(super) fn check(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<(), MatmulError> {
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

pub(super) fn rows(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32]) {
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
