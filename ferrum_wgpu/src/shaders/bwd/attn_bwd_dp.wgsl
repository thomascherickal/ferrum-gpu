// dP[b,h,i,j] = dO[i] . V[j] for j <= i (0 beyond the diagonal), one invocation per cell.
// Written into the `ds` scratch, which later kernels turn into dS in place.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> ds: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.batch * u.heads * u.t * u.t) { return; }
    let j = idx % u.t;
    let r = idx / u.t;
    let i = r % u.t;
    let bh = r / u.t;
    let h = bh % u.heads;
    let b = bh / u.heads;
    if (j > i) { ds[idx] = 0.0; return; }
    let c = u.heads * u.hd;
    var dp = 0.0;
    for (var d = 0u; d < u.hd; d = d + 1u) {
        dp = dp + dout[(b * u.t + i) * c + h * u.hd + d] * v[(b * u.t + j) * c + h * u.hd + d];
    }
    ds[idx] = dp;
}
