// Single-query grouped-query attention probabilities over a KV cache laid out [ctx, kv_heads*hd].
// One invocation per query head; positions 0..len are visible; probs is [heads, ctx].
struct U { heads: u32, kv_heads: u32, hd: u32, len: u32, ctx: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let h = gid(g);
    if (h >= u.heads) { return; }
    let kvd = u.kv_heads * u.hd;
    let koff = (h / (u.heads / u.kv_heads)) * u.hd;
    let scale = 1.0 / sqrt(f32(u.hd));
    let base = h * u.ctx;
    var mx = -3.4e38;
    for (var j = 0u; j < u.len; j = j + 1u) {
        var dot = 0.0;
        for (var d = 0u; d < u.hd; d = d + 1u) {
            dot = dot + q[h * u.hd + d] * kc[j * kvd + koff + d];
        }
        let s = dot * scale;
        probs[base + j] = s;
        mx = max(mx, s);
    }
    var sum = 0.0;
    for (var j = 0u; j < u.len; j = j + 1u) {
        let e = exp(probs[base + j] - mx);
        probs[base + j] = e;
        sum = sum + e;
    }
    for (var j = 0u; j < u.len; j = j + 1u) { probs[base + j] = probs[base + j] / sum; }
}
