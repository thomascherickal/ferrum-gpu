// out[r, d] = table[ids[r], d]
struct U { rows: u32, c: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.rows * u.c) { return; }
    out[idx] = table[ids[idx / u.c] * u.c + idx % u.c];
}
