// One invocation per causal (batch, head h, query i, key j) cell: OUT = (A[i, head h] .
// B[j, kv head of h]) * SCALE for j <= i and 0 beyond the diagonal, d summed in ascending order.
// Query head h reads kv head h / (heads / KV); GPT-2 kernels pass KV=u.heads (one kv head each).
@group(0) @binding(0) var<storage, read> {{A}}: array<f32>;
@group(0) @binding(1) var<storage, read> {{B}}: array<f32>;
@group(0) @binding(2) var<storage, read_write> {{OUT}}: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.batch * u.heads * u.t * u.t) { return; }
    let j = idx % u.t; let i = (idx / u.t) % u.t;
    let bh = idx / (u.t * u.t);
    let h = bh % u.heads; let b = bh / u.heads;
    if (j > i) { {{OUT}}[idx] = 0.0; return; }
    let qd = u.heads * u.hd;
    let kvd = {{KV}} * u.hd;
    let koff = (h / (u.heads / {{KV}})) * u.hd;
    var dot = 0.0;
    for (var d = 0u; d < u.hd; d = d + 1u) {
        dot = dot + {{A}}[(b * u.t + i) * qd + h * u.hd + d] * {{B}}[(b * u.t + j) * kvd + koff + d];
    }
    {{OUT}}[idx] = dot * {{SCALE}};
}
