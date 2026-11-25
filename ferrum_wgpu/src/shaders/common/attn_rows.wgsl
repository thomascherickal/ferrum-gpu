// One invocation per output element (row (b, i), column col of query head h):
// OUT = sum over j <= i, ascending, of P[b, h, i, j] * V[j, kv head of h]. Query head h reads kv
// head h / (heads / KV); GPT-2 kernels pass KV=u.heads (one kv head each).
@group(0) @binding(0) var<storage, read> {{P}}: array<f32>;
@group(0) @binding(1) var<storage, read> {{V}}: array<f32>;
@group(0) @binding(2) var<storage, read_write> {{OUT}}: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let qd = u.heads * u.hd;
    let kvd = {{KV}} * u.hd;
    let idx = gid(g);
    if (idx >= u.batch * u.t * qd) { return; }
    let col = idx % qd;
    let row = idx / qd;
    let b = row / u.t;
    let i = row % u.t;
    let h = col / u.hd;
    let vcol = (h / (u.heads / {{KV}})) * u.hd + col % u.hd;
    let pbase = ((b * u.heads + h) * u.t + i) * u.t;
    var sum = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        sum = sum + {{P}}[pbase + j] * {{V}}[(b * u.t + j) * kvd + vcol];
    }
    {{OUT}}[idx] = sum;
}
