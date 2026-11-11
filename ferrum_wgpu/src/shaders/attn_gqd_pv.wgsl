// out[i] = sum_{j<len} probs[head(i), j] * vc[j, kv column of i]; one invocation per output element.
struct U { heads: u32, kv_heads: u32, hd: u32, len: u32, ctx: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> vc: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gid(g);
    if (i >= u.heads * u.hd) { return; }
    let kvd = u.kv_heads * u.hd;
    let h = i / u.hd;
    let vcol = (h / (u.heads / u.kv_heads)) * u.hd + i % u.hd;
    var sum = 0.0;
    for (var j = 0u; j < u.len; j = j + 1u) {
        sum = sum + probs[h * u.ctx + j] * vc[j * kvd + vcol];
    }
    out[i] = sum;
}
