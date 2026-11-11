// dV[(b,j), kvh*hd+d] = sum over the query heads h of the group, then i >= j (ascending), of
// P[b,h,i,j] * dO[(b,i), h*hd+d]. One invocation per output element.
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read_write> dv: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let kvd = u.kv_heads * u.hd;
    let qd = u.heads * u.hd;
    let idx = gid(g);
    if (idx >= u.batch * u.t * kvd) { return; }
    let col = idx % kvd;
    let row = idx / kvd;
    let b = row / u.t;
    let j = row % u.t;
    let kvh = col / u.hd;
    let d = col % u.hd;
    let group = u.heads / u.kv_heads;
    var s = 0.0;
    for (var gi = 0u; gi < group; gi = gi + 1u) {
        let h = kvh * group + gi;
        for (var i = j; i < u.t; i = i + 1u) {
            s = s + probs[((b * u.heads + h) * u.t + i) * u.t + j] * dout[(b * u.t + i) * qd + h * u.hd + d];
        }
    }
    dv[idx] = s;
}
