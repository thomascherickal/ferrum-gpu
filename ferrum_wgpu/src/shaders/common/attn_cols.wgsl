// One invocation per output element (row (b, j), column of kv head kvh): OUT = the sum, over the
// query heads h of kvh in order and then i >= j ascending, of P[b, h, i, j] * X[i, head h].
// GPT-2 kernels pass KV=u.heads, so each kv head has one query head.
@group(0) @binding(0) var<storage, read> {{P}}: array<f32>;
@group(0) @binding(1) var<storage, read> {{X}}: array<f32>;
@group(0) @binding(2) var<storage, read_write> {{OUT}}: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let kvd = {{KV}} * u.hd;
    let qd = u.heads * u.hd;
    let idx = gid(g);
    if (idx >= u.batch * u.t * kvd) { return; }
    let col = idx % kvd;
    let row = idx / kvd;
    let b = row / u.t;
    let j = row % u.t;
    let kvh = col / u.hd;
    let d = col % u.hd;
    let group = u.heads / {{KV}};
    var s = 0.0;
    for (var gi = 0u; gi < group; gi = gi + 1u) {
        let h = kvh * group + gi;
        for (var i = j; i < u.t; i = i + 1u) {
            s = s + {{P}}[((b * u.heads + h) * u.t + i) * u.t + j] * {{X}}[(b * u.t + i) * qd + h * u.hd + d];
        }
    }
    {{OUT}}[idx] = s;
}
