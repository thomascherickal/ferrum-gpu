// Blocked4 with optional transposed operands, for the weight and input gradients.
// flags bit 0: A is stored [k, m] (use A^T); bit 1: B is stored [n, k] (use B^T).
// Same tiling and ascending-k accumulation as matmul_blocked4.wgsl.
struct Dims { m: u32, k: u32, n: u32, flags: u32 }

fn load_a(row: u32, kk: u32) -> f32 {
    if (row >= d.m || kk >= d.k) { return 0.0; }
    if ((d.flags & 1u) != 0u) { return a[kk * d.m + row]; }
    return a[row * d.k + kk];
}

fn load_b(kk: u32, col: u32) -> f32 {
    if (kk >= d.k || col >= d.n) { return 0.0; }
    if ((d.flags & 2u) != 0u) { return b[col * d.k + kk]; }
    return b[kk * d.n + col];
}

//! use common/blocked4
