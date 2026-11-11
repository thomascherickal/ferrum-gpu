// out[0] = clip scale, out[1] = global gradient norm. Sums the partials in ascending order.
// scale = max_norm / norm when max_norm > 0 and the norm is finite and above the cap, else 1.
struct U { total: u32, max_bits: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    for (var i = 0u; i < u.total; i = i + 1u) { s = s + part[i]; }
    let norm = sqrt(s);
    let mx = bitcast<f32>(u.max_bits);
    var scale = 1.0;
    if (mx > 0.0 && norm == norm && norm < 3.0e38 && norm > mx) { scale = mx / norm; }
    out[0] = scale;
    out[1] = norm;
}
