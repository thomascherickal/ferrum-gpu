// out[i] = sum_{j<len} probs[head(i), j] * vc[j, i]; one invocation per output element.
struct U { heads: u32, hd: u32, len: u32, ctx: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> vc: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = u.heads * u.hd;
    let i = gid(g);
    if (i >= c) { return; }
    let h = i / u.hd;
    var sum = 0.0;
    for (var j = 0u; j < u.len; j = j + 1u) {
        sum = sum + probs[h * u.ctx + j] * vc[j * c + i];
    }
    out[i] = sum;
}
