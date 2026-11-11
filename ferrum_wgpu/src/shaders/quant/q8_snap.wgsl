// Snap a tensor onto its Q8_0 values (quantise each block of 32, then dequantise), exactly as the
// CPU reference does: d = amax / 127, q = round-half-away(x / d) clamped to i8, the scale rounded
// to f16 (nearest even) by integer bit operations, snapped = q * f16(d).
// One invocation per block. mode 0: the block is 32 consecutive elements. mode 1: the block runs
// down a column of an [in, out] matrix (stride = out); `cols` is `out`.
struct U { blocks: u32, mode: u32, cols: u32, p0: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

fn f32_to_f16_bits(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let sign = (b >> 16u) & 0x8000u;
    let exp = i32((b >> 23u) & 0xffu);
    let man = b & 0x007fffffu;
    if (exp == 0xff) { return sign | 0x7c00u | select(0u, 0x200u, man != 0u); }
    let e = exp - 127 + 15;
    if (e >= 0x1f) { return sign | 0x7c00u; }
    if (e <= 0) {
        if (e < -10) { return sign; }
        let m = man | 0x00800000u;
        let shift = u32(14 - e);
        let half = m >> shift;
        let rem = m & ((1u << shift) - 1u);
        let mid = 1u << (shift - 1u);
        let up = rem > mid || (rem == mid && (half & 1u) == 1u);
        return sign | (half + select(0u, 1u, up));
    }
    let half = (u32(e) << 10u) | (man >> 13u);
    let rem = man & 0x1fffu;
    let up = rem > 0x1000u || (rem == 0x1000u && (half & 1u) == 1u);
    return sign | (half + select(0u, 1u, up));
}

fn f16_bits_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) << 31u;
    let exp = (h >> 10u) & 0x1fu;
    let man = h & 0x3ffu;
    if (exp == 0u) {
        if (man == 0u) { return bitcast<f32>(sign); }
        let shift = countLeadingZeros(man) - 21u;
        let m = (man << shift) & 0x3ffu;
        let e = 127u - 15u + 1u - shift;
        return bitcast<f32>(sign | (e << 23u) | (m << 13u));
    }
    if (exp == 0x1fu) { return bitcast<f32>(sign | 0x7f800000u | (man << 13u)); }
    return bitcast<f32>(sign | ((exp + 127u - 15u) << 23u) | (man << 13u));
}

// Round half away from zero, exact: the fractional part of |x| is computed without rounding.
fn round_away(x: f32) -> f32 {
    let a = abs(x);
    let t = trunc(a);
    let r = t + select(0.0, 1.0, a - t >= 0.5);
    return select(r, -r, x < 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let blk = gid(g);
    if (blk >= u.blocks) { return; }
    var base = blk * 32u;
    var stride = 1u;
    if (u.mode == 1u) {
        base = (blk / u.cols) * 32u * u.cols + (blk % u.cols);
        stride = u.cols;
    }
    var amax = 0.0;
    for (var k = 0u; k < 32u; k = k + 1u) { amax = max(amax, abs(src[base + k * stride])); }
    let d = amax / 127.0;
    var id = 0.0;
    if (d != 0.0) { id = 1.0 / d; }
    let dq = f16_bits_to_f32(f32_to_f16_bits(d));
    for (var k = 0u; k < 32u; k = k + 1u) {
        // The CPU quantises to an i8, so a value that rounds to zero comes out as +0. Some
        // compilers fold f32(i32(x)) back into a float truncation, which keeps -0, so the zero
        // case writes a literal +0.0 instead of relying on that conversion.
        let q = i32(clamp(round_away(src[base + k * stride] * id), -128.0, 127.0));
        var o = 0.0;
        if (q != 0) { o = f32(q) * dq; }
        dst[base + k * stride] = o;
    }
}
