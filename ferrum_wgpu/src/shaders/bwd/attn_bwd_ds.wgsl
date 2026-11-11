// In place: ds <- P * (dP - dot) / sqrt(hd) for j <= i, 0 beyond the diagonal. One invocation per cell.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> dot: array<f32>;
@group(0) @binding(2) var<storage, read_write> ds: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.batch * u.heads * u.t * u.t) { return; }
    let j = idx % u.t;
    let r = idx / u.t;
    let i = r % u.t;
    if (j > i) { ds[idx] = 0.0; return; }
    ds[idx] = probs[idx] * (ds[idx] - dot[r]) * (1.0 / sqrt(f32(u.hd)));
}
