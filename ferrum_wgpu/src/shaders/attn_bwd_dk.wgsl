// dK[(b,j), col] = sum_{i>=j} dS[i,j] * Q[(b,i), col], ascending i; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> dot: array<f32>;
@group(0) @binding(5) var<storage, read_write> dk: array<f32>;
@group(0) @binding(6) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = u.heads * u.hd;
    let idx = g.x;
    if (idx >= u.batch * u.t * c) { return; }
    let col = idx % c;
    let row = idx / c;
    let b = row / u.t;
    let j = row % u.t;
    let h = col / u.hd;
    let scale = 1.0 / sqrt(f32(u.hd));
    var s = 0.0;
    for (var i = j; i < u.t; i = i + 1u) {
        let r = (b * u.heads + h) * u.t + i;
        var dp = 0.0;
        for (var d = 0u; d < u.hd; d = d + 1u) {
            dp = dp + dout[(b * u.t + i) * c + h * u.hd + d] * v[row * c + h * u.hd + d];
        }
        let ds = probs[r * u.t + j] * (dp - dot[r]) * scale;
        s = s + ds * q[(b * u.t + i) * c + col];
    }
    dk[idx] = s;
}
