// dst[r][c] = (add ? dst[r][c] : 0) + src[c][r]: `dst` is [rows, cols] row-major and `src` is its
// transpose, [cols, rows]. One thread per element of `dst`; `dst` is read_write and `src` read-only,
// so no buffer is bound twice. Used by tied embedding/head training: the head is the transposed
// embedding, and the head's gradient is added into the embedding's (transposed).
struct U { rows: u32, cols: u32, add: u32, p0: u32 }
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = gid(id);
    if (i >= u.rows * u.cols) { return; }
    let r = i / u.cols;
    let c = i % u.cols;
    var base = 0.0;
    if (u.add != 0u) { base = dst[i]; }
    dst[i] = base + src[c * u.rows + r];
}
