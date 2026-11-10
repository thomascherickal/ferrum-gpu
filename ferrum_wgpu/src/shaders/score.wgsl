// Per row: out[2r] = log-probability of the target, out[2r+1] = 1.0 when the target is the first argmax.
struct U { rows: u32, vocab: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = g.x;
    if (r >= u.rows) { return; }
    let base = r * u.vocab;
    var mx = -3.4e38;
    var first = 0u;
    for (var j = 0u; j < u.vocab; j = j + 1u) {
        let v = logits[base + j];
        if (v > mx) { mx = v; first = j; }
    }
    var sum = 0.0;
    for (var j = 0u; j < u.vocab; j = j + 1u) { sum = sum + exp(logits[base + j] - mx); }
    let t = targets[r];
    out[2u * r] = logits[base + t] - mx - log(sum);
    out[2u * r + 1u] = select(0.0, 1.0, first == t);
}
