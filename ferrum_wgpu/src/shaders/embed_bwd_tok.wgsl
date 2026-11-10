// dtok[v, d] = sum of dx[r, d] over the rows r whose token is v, in ascending row order.
// `order` lists row indices sorted by token (stable), `start` is the CSR row_start (vocab + 1).
// No atomics: one invocation per output element, so the sum order is fixed.
struct U { vocab: u32, c: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> dx: array<f32>;
@group(0) @binding(1) var<storage, read> start: array<u32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> dtok: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.vocab * u.c) { return; }
    let tok = idx / u.c;
    let d = idx % u.c;
    var s = 0.0;
    for (var k = start[tok]; k < start[tok + 1u]; k = k + 1u) {
        s = s + dx[order[k] * u.c + d];
    }
    dtok[idx] = s;
}
