// dK[(b,j), col] = sum_{i>=j} dS[b,h,i,j] * Q[(b,i), col], ascending i; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> ds: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> dk: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = u.heads * u.hd;
    let idx = gid(g);
    if (idx >= u.batch * u.t * c) { return; }
    let col = idx % c;
    let row = idx / c;
    let b = row / u.t;
    let j = row % u.t;
    let h = col / u.hd;
    var s = 0.0;
    for (var i = j; i < u.t; i = i + 1u) {
        s = s + ds[((b * u.heads + h) * u.t + i) * u.t + j] * q[(b * u.t + i) * c + col];
    }
    dk[idx] = s;
}
