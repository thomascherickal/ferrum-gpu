// 64x64 output tile per workgroup (16x16 invocations, 4x4 outputs each), K tile 16,
// ascending-k accumulation per output element; both operands stored row-major.
struct Dims { m: u32, k: u32, n: u32, pad: u32 }

fn load_a(row: u32, kk: u32) -> f32 {
    if (row >= d.m || kk >= d.k) { return 0.0; }
    return a[row * d.k + kk];
}

fn load_b(kk: u32, col: u32) -> f32 {
    if (kk >= d.k || col >= d.n) { return 0.0; }
    return b[kk * d.n + col];
}

//! use common/blocked4
