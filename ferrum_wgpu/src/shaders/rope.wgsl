// In-place RoPE. One invocation per (row, head, frequency pair); cos/sin come from host-built
// tables (never sin/cos in a shader), row r reading table row table_offset + r. neox = 0 rotates
// adjacent pairs (2i, 2i+1), neox = 1 rotates split halves (i, i + rope_dim/2). Channels at or
// past rope_dim are untouched.
struct U { rows: u32, heads: u32, hd: u32, rope_dim: u32, neox: u32, table_offset: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let half = u.rope_dim / 2u;
    let idx = gid(g);
    if (idx >= u.rows * u.heads * half) { return; }
    let i = idx % half;
    let rh = idx / half;
    let r = rh / u.heads;
    let off = rh * u.hd;
    var ia = off + 2u * i;
    var ib = ia + 1u;
    if (u.neox == 1u) { ia = off + i; ib = off + i + half; }
    let c = cos_t[(u.table_offset + r) * half + i];
    let s = sin_t[(u.table_offset + r) * half + i];
    let a = x[ia];
    let b = x[ib];
    x[ia] = a * c - b * s;
    x[ib] = a * s + b * c;
}
