// One partial sum of squares per workgroup (4096 elements): each of 256 invocations sums 16
// strided elements in ascending order, then a fixed pairwise tree combines them. The result
// depends only on the data, never on scheduling.
struct U { n: u32, off: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let base = wg.x * 4096u;
    var s = 0.0;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let i = base + k * 256u + l.x;
        if (i < u.n) { s = s + g[i] * g[i]; }
    }
    sh[l.x] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (l.x < stride) { sh[l.x] = sh[l.x] + sh[l.x + stride]; }
        workgroupBarrier();
    }
    if (l.x == 0u) { part[u.off + wg.x] = sh[0]; }
}
