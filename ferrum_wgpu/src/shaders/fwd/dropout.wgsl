// In-place inverted dropout: x[i] = 0 if unit (offset + i) is dropped, else x[i] * scale.
// The mask is a pure function of (seed, block, unit index), the same 32-bit hash the CPU uses
// (ferrum_ref::dropout::uniform): u32 arithmetic wraps here exactly like Rust's wrapping_*, and
// (h >> 8) -> f32 -> / 2^24 is exact, so the GPU and CPU masks are bit-identical. The same
// kernel masks the activations in the forward pass and their gradient in the backward pass.
struct U { n: u32, rate: u32, scale: u32, seed_lo: u32, seed_hi: u32, block: u32, offset: u32, p0: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

// lowbias32
fn mix(v: u32) -> u32 {
    var z = v;
    z = z ^ (z >> 16u);
    z = z * 0x7feb352du;
    z = z ^ (z >> 15u);
    z = z * 0x846ca68bu;
    z = z ^ (z >> 16u);
    return z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = gid(id);
    if (i >= u.n) { return; }
    let h = mix(mix(mix(mix(u.seed_lo) ^ u.seed_hi) ^ u.block) ^ (u.offset + i));
    let r = f32(h >> 8u) / 16777216.0;
    if (r < bitcast<f32>(u.rate)) {
        x[i] = 0.0;
    } else {
        x[i] = x[i] * bitcast<f32>(u.scale);
    }
}
