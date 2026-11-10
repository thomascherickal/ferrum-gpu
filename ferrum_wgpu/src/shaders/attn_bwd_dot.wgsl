// dot[b,h,i] = sum_{j<=i} P[i,j] * (dO[i] . V[j]); one invocation per (batch, head, query) row.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> dot: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.batch * u.heads * u.t) { return; }
    let i = r % u.t;
    let bh = r / u.t;
    let h = bh % u.heads;
    let b = bh / u.heads;
    let c = u.heads * u.hd;
    var s = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        var dp = 0.0;
        for (var d = 0u; d < u.hd; d = d + 1u) {
            dp = dp + dout[(b * u.t + i) * c + h * u.hd + d] * v[(b * u.t + j) * c + h * u.hd + d];
        }
        s = s + probs[r * u.t + j] * dp;
    }
    dot[r] = s;
}
