use super::rig::Rig;
use ferrum_wgpu::{ops, ops_bwd as bwd};

/// LayerNorm forward and backward, GELU, bias, residual add.
pub fn rows(rig: &mut Rig) {
    let (r, c, h, layers) = (rig.d.rows(), rig.d.c, rig.d.h, rig.d.layers);
    let (x, gain, shift) = (rig.buf(r * c), rig.buf(c), rig.buf(c));
    let (y, st) = (rig.zeros(r * c), rig.zeros(2 * r));
    rig.record("layernorm forward", 2 * layers + 1, |g| {
        ops::layernorm(g, &x, &gain, &shift, &y, (r, c)).unwrap()
    });
    let (dx, dg, db) = (rig.zeros(r * c), rig.zeros(c), rig.zeros(c));
    rig.record("layernorm backward", 2 * layers + 1, |g| {
        bwd::layernorm_bwd(g, (&x, &gain, &y), (&dx, &dg, &db), &st, (r, c)).unwrap()
    });
    let (f1, act) = (rig.buf(r * h), rig.zeros(r * h));
    rig.record("gelu forward", layers, |g| ops::gelu(g, &f1, &act).unwrap());
    let (da, df) = (rig.buf(r * h), rig.zeros(r * h));
    rig.record("gelu backward", layers, |g| {
        bwd::gelu_bwd(g, &da, &f1, &df).unwrap()
    });
    let (bias, yb) = (rig.buf(c), rig.zeros(r * c));
    rig.record("bias add [1024x512]", 5 * layers, |g| {
        ops::add_bias(g, &x, &bias, &yb).unwrap()
    });
    let (xa, xb, xc) = (rig.buf(r * c), rig.buf(r * c), rig.zeros(r * c));
    rig.record("residual add [1024x512]", 4 * layers + 2, |g| {
        ops::add(g, &xa, &xb, &xc).unwrap()
    });
    // Column sums (bias gradients), the largest being the FFN-up bias and the junk head bias.
    for (name, cols, per) in [
        ("colsum [1024x512] (bias grad)", c, 5 * layers),
        ("colsum [1024x2048] (ffn bias grad)", h, layers),
        ("colsum [1024x4096] (head, unused result)", rig.d.v, 1),
    ] {
        let (xin, out) = (rig.buf(r * cols), rig.zeros(cols));
        rig.record(name, per, |g| {
            bwd::colsum(g, &xin, &out, (r, cols)).unwrap()
        });
    }
}
