// One invocation per query head h of a single query over a KV cache laid out [ctx, KV*hd]:
// probs[h, 0..len] = softmax of the scaled dots with the cached keys of kv head h / (heads / KV)
// (positions 0..len visible; the rest of the row is not written). GPT-2 passes KV=u.heads.
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let h = gid(g);
    if (h >= u.heads) { return; }
    let kvd = {{KV}} * u.hd;
    let koff = (h / (u.heads / {{KV}})) * u.hd;
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
