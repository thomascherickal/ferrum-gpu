// acc[i] = (first ? 0 : acc[i]) + w * g[i]: gradient accumulation across micro-batches. `acc` is
// read_write and `g` read-only, so no buffer is bound twice. Chunks are added in order.
struct U { n: u32, w: u32, first: u32, p0: u32 }
@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = gid(id);
    if (i >= u.n) { return; }
    var base = acc[i];
    if (u.first != 0u) { base = 0.0; }
    acc[i] = base + bitcast<f32>(u.w) * g[i];
}
