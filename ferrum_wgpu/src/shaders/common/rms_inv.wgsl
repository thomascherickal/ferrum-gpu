// 1 / sqrt(mean(x^2) + eps) over the row starting at `base`, summed serially in ascending order
// (the CPU reference order); eps arrives as f32 bits in the uniform.
fn rms_inv(base: u32) -> f32 {
    var ss = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) { ss = ss + x[base + c] * x[base + c]; }
    return 1.0 / sqrt(ss / f32(u.cols) + bitcast<f32>(u.eps));
}
