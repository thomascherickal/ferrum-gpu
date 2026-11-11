// dQ[(b,i), h*hd+d] = sum_{j<=i} dS[b,h,i,j] * K[(b,j), kvh*hd+d], ascending j; one invocation
// per output element.
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> ds: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> dq: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let qd = u.heads * u.hd;
    let kvd = u.kv_heads * u.hd;
    let idx = gid(g);
    if (idx >= u.batch * u.t * qd) { return; }
    let col = idx % qd;
    let row = idx / qd;
    let b = row / u.t;
    let i = row % u.t;
    let h = col / u.hd;
    let kcol = (h / (u.heads / u.kv_heads)) * u.hd + col % u.hd;
    let r = (b * u.heads + h) * u.t + i;
    var s = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        s = s + ds[r * u.t + j] * k[(b * u.t + j) * kvd + kcol];
    }
    dq[idx] = s;
}
