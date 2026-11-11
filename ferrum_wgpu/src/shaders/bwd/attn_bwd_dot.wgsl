// dot[b,h,i] = sum_{j<=i} P[i,j] * dP[i,j] (ascending j); one invocation per row. `ds` holds dP.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> ds: array<f32>;
@group(0) @binding(2) var<storage, read_write> dot: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.batch * u.heads * u.t) { return; }
    let i = r % u.t;
    var s = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        s = s + probs[r * u.t + j] * ds[r * u.t + j];
    }
    dot[r] = s;
}
