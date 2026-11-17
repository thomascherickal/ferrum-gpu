// y[col] = sum_k x[k] * W[col, k] for one row (decode). One workgroup of 32 invocations per output
// column: invocation e takes element e of every 32-block along k, so a column is read by 32
// invocations at once instead of one invocation walking all of k, and the 32 partial sums are
// added in a tree. Same packed layout and dequantised values as qlinear.wgsl; only the order of
// the sum differs.  kind 0 Q8_0 (9 words per record), 1 Q4_0 (5), 2 F16 (16, no scale).
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> red: array<f32, 32>;

fn record_words(kind: u32) -> u32 {
    if (kind == 0u) { return 9u; }
    if (kind == 1u) { return 5u; }
    return 16u;
}

fn deq(kind: u32, rec: u32, e: u32) -> f32 {
    if (kind == 0u) {
        let d = bitcast<f32>(w[rec]);
        let q = bitcast<i32>(w[rec + 1u + e / 4u] << (24u - 8u * (e % 4u))) >> 24u;
        return d * f32(q);
    }
    if (kind == 1u) {
        let d = bitcast<f32>(w[rec]);
        let j = e % 16u;
        let byte = (w[rec + 1u + j / 4u] >> (8u * (j % 4u))) & 0xffu;
        let nib = select(byte >> 4u, byte & 15u, e < 16u);
        return d * (f32(nib) - 8.0);
    }
    let v = unpack2x16float(w[rec + e / 2u]);
    return select(v.y, v.x, e % 2u == 0u);
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let col = wg.x + wg.y * 65535u;
    let e = l.x;
    var acc = 0.0;
    if (col < u.n) {
        let rw = record_words(u.kind);
        for (var b = 0u; b < u.k / 32u; b = b + 1u) {
            acc = acc + x[b * 32u + e] * deq(u.kind, (b * u.n + col) * rw, e);
        }
    }
    red[e] = acc;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s / 2u) {
        if (e < s) { red[e] = red[e] + red[e + s]; }
        workgroupBarrier();
    }
    if (e == 0u && col < u.n) { y[col] = red[0]; }
}
