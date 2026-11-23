use super::*;

pub(crate) fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<(), GgufError> {
    if got == want {
        Ok(())
    } else {
        Err(err(format!(
            "tensor '{name}' has {got} elements, expected {want}"
        )))
    }
}

/// Read a tensor, check its element count, return its f32 data.
pub(crate) fn load(g: &Gguf, name: &str, want: usize) -> Result<Vec<f32>, GgufError> {
    let v = g.tensor_f32(name)?;
    expect_len(name, v.len(), want)?;
    Ok(v)
}

/// Load a stored `[out, in]` matrix as our `[in, out]`.
pub(crate) fn load_matrix(
    g: &Gguf,
    name: &str,
    inp: usize,
    out: usize,
) -> Result<Vec<f32>, GgufError> {
    Ok(transpose(&load(g, name, inp * out)?, out, inp))
}
