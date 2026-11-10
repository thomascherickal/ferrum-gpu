// AdamW update of one tensor, same operation order as the CPU reference:
//   g = grad * clip_scale; m = b1 m + (1-b1) g; v = b2 v + (1-b2) g g;
//   update = lr * (m / bc1) / (sqrt(v / bc2) + eps) [+ lr * wd * p]; p -= update.
// Hyperparameters arrive as f32 bit patterns; bias corrections and lr are computed on the host.
struct U {
    n: u32, lr: u32, b1: u32, b2: u32,
    eps: u32, wd: u32, bc1: u32, bc2: u32,
}
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read> scale: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gid(g);
    if (i >= u.n) { return; }
    let lr = bitcast<f32>(u.lr);
    let b1 = bitcast<f32>(u.b1);
    let b2 = bitcast<f32>(u.b2);
    let wd = bitcast<f32>(u.wd);
    let gr = grad[i] * scale[0];
    let mi = b1 * m[i] + (1.0 - b1) * gr;
    let vi = b2 * v[i] + (1.0 - b2) * gr * gr;
    m[i] = mi;
    v[i] = vi;
    let mh = mi / bitcast<f32>(u.bc1);
    let vh = vi / bitcast<f32>(u.bc2);
    var update = lr * mh / (sqrt(vh) + bitcast<f32>(u.eps));
    if (wd != 0.0) { update = update + lr * wd * p[i]; }
    p[i] = p[i] - update;
}
