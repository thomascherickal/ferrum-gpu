// Softmax cross-entropy per row: row loss and gradient (p - onehot) / rows. One invocation per row.
struct U { rows: u32, vocab: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> row_loss: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = g.x;
    if (r >= u.rows) { return; }
    let base = r * u.vocab;
    var mx = -3.4e38;
    for (var j = 0u; j < u.vocab; j = j + 1u) { mx = max(mx, logits[base + j]); }
    var sum = 0.0;
    for (var j = 0u; j < u.vocab; j = j + 1u) {
        let e = exp(logits[base + j] - mx);
        grad[base + j] = e;
        sum = sum + e;
    }
    let inv = 1.0 / f32(u.rows);
    let t = targets[r];
    for (var j = 0u; j < u.vocab; j = j + 1u) {
        var p = grad[base + j] / sum;
        if (j == t) {
            row_loss[r] = -log(max(p, 1e-12));
            p = p - 1.0;
        }
        grad[base + j] = p * inv;
    }
}
