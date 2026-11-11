// Grouped-query causal scores, one invocation per (batch, head, query, key) cell. Query head h
// reads KV head h / (heads / kv_heads). Cells with key > query get 0 (and stay 0 after softmax).
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.batch * u.heads * u.t * u.t) { return; }
    let j = idx % u.t;
    let r = idx / u.t;
    let i = r % u.t;
    let bh = r / u.t;
    let h = bh % u.heads;
    let b = bh / u.heads;
    if (j > i) { probs[idx] = 0.0; return; }
    let qd = u.heads * u.hd;
    let kvd = u.kv_heads * u.hd;
    let koff = (h / (u.heads / u.kv_heads)) * u.hd;
    var dot = 0.0;
    for (var d = 0u; d < u.hd; d = d + 1u) {
        dot = dot + q[(b * u.t + i) * qd + h * u.hd + d] * k[(b * u.t + j) * kvd + koff + d];
    }
    probs[idx] = dot * (1.0 / sqrt(f32(u.hd)));
}
