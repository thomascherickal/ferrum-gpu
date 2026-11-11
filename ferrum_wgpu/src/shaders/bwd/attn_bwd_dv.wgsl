// dV[(b,j), col] = sum_{i>=j} P[b,h,i,j] * dO[(b,i), col], ascending i; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read_write> dv: array<f32>;
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
        s = s + probs[((b * u.heads + h) * u.t + i) * u.t + j] * dout[(b * u.t + i) * c + col];
    }
    dv[idx] = s;
}
