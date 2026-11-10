// dQ[(b,i), col] = sum_{j<=i} dS[i,j] * K[(b,j), col] with
// dS[i,j] = P[i,j] * (dO[i].V[j] - dot[i]) / sqrt(hd); ascending j; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> k: array<f32>;
@group(0) @binding(4) var<storage, read> dot: array<f32>;
@group(0) @binding(5) var<storage, read_write> dq: array<f32>;
@group(0) @binding(6) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = u.heads * u.hd;
    let idx = gid(g);
    if (idx >= u.batch * u.t * c) { return; }
    let col = idx % c;
    let row = idx / c;
    let b = row / u.t;
    let i = row % u.t;
    let h = col / u.hd;
    let scale = 1.0 / sqrt(f32(u.hd));
    let r = (b * u.heads + h) * u.t + i;
    var s = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        var dp = 0.0;
        for (var d = 0u; d < u.hd; d = d + 1u) {
            dp = dp + dout[row * c + h * u.hd + d] * v[(b * u.t + j) * c + h * u.hd + d];
        }
        let ds = probs[r * u.t + j] * (dp - dot[r]) * scale;
        s = s + ds * k[(b * u.t + j) * c + col];
    }
    dq[idx] = s;
}
