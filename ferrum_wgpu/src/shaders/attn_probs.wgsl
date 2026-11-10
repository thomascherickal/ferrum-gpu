// Causal attention probabilities. One invocation per (batch, head, query position) row:
// scaled dot products for j <= i, max-subtracted softmax; masked entries are exactly 0
// (the CPU's exp(-1e9 - max) underflows to 0).
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.batch * u.heads * u.t) { return; }
    let i = r % u.t;
    let bh = r / u.t;
    let h = bh % u.heads;
    let b = bh / u.heads;
    let c = u.heads * u.hd;
    let scale = 1.0 / sqrt(f32(u.hd));
    let base = r * u.t;
    var mx = -3.4e38;
    for (var j = 0u; j <= i; j = j + 1u) {
        var dot = 0.0;
        for (var d = 0u; d < u.hd; d = d + 1u) {
            dot = dot + q[(b * u.t + i) * c + h * u.hd + d] * k[(b * u.t + j) * c + h * u.hd + d];
        }
        let s = dot * scale;
        probs[base + j] = s;
        mx = max(mx, s);
    }
    var sum = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        let e = exp(probs[base + j] - mx);
        probs[base + j] = e;
        sum = sum + e;
    }
    for (var j = 0u; j <= i; j = j + 1u) { probs[base + j] = probs[base + j] / sum; }
    for (var j = i + 1u; j < u.t; j = j + 1u) { probs[base + j] = 0.0; }
}
