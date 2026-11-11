// Scaled attention scores, one invocation per (batch, head, query, key) cell so the whole
// t x t matrix is computed in parallel. Cells with key > query (the causal mask) get 0 here and
// stay exactly 0 after the softmax. Each dot product sums d in ascending order.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
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
    let c = u.heads * u.hd;
    var dot = 0.0;
    for (var d = 0u; d < u.hd; d = d + 1u) {
        dot = dot + q[(b * u.t + i) * c + h * u.hd + d] * k[(b * u.t + j) * c + h * u.hd + d];
    }
    probs[idx] = dot * (1.0 / sqrt(f32(u.hd)));
}
