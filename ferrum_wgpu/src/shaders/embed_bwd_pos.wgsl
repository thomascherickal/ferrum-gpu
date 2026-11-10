// dpos[p, d] = sum of dx[r, d] over rows r with r % t == p, ascending r; positions >= t get 0.
struct U { rows: u32, c: u32, t: u32, ctx: u32 }
@group(0) @binding(0) var<storage, read> dx: array<f32>;
@group(0) @binding(1) var<storage, read_write> dpos: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= u.ctx * u.c) { return; }
    let p = idx / u.c;
    let d = idx % u.c;
    var s = 0.0;
    if (p < u.t) {
        for (var r = p; r < u.rows; r = r + u.t) { s = s + dx[r * u.c + d]; }
    }
    dpos[idx] = s;
}
