// out[r, d] = tok[ids[r], d] + pos[r % t, d]
struct U { rows: u32, c: u32, t: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> tok: array<f32>;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= u.rows * u.c) { return; }
    let r = idx / u.c;
    let d = idx % u.c;
    out[idx] = tok[ids[r] * u.c + d] + pos[(r % u.t) * u.c + d];
}
