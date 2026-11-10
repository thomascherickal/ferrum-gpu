// In-place causal softmax over each (batch, head, query) row of scores: max-subtracted,
// ascending sums, entries past the diagonal left at exactly 0. One invocation per row.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.batch * u.heads * u.t) { return; }
    let i = r % u.t;
    let base = r * u.t;
    var mx = -3.4e38;
    for (var j = 0u; j <= i; j = j + 1u) { mx = max(mx, probs[base + j]); }
    var sum = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        let e = exp(probs[base + j] - mx);
        probs[base + j] = e;
        sum = sum + e;
    }
    for (var j = 0u; j <= i; j = j + 1u) { probs[base + j] = probs[base + j] / sum; }
}
