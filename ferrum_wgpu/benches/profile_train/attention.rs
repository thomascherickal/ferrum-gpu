use super::rig::Rig;
use ferrum_wgpu::{ops, ops_bwd as bwd};

/// Attention forward and backward.
pub(crate) fn rows(rig: &mut Rig) {
    let (b, t, heads, layers) = (rig.d.b, rig.d.t, rig.d.heads, rig.d.layers);
    let hd = rig.d.c / heads;
    let n = rig.d.rows() * rig.d.c;
    let (q, k, v) = (rig.buf(n), rig.buf(n), rig.buf(n));
    let (probs, ao) = (rig.zeros(b * heads * t * t), rig.zeros(n));
    rig.record("attention forward", layers, |g| {
        ops::attention(g, (&q, &k, &v), (&probs, &ao), (b, t, heads, hd)).unwrap()
    });
    let (dout, dq, dk, dv) = (rig.buf(n), rig.zeros(n), rig.zeros(n), rig.zeros(n));
    let (dot, dsc) = (rig.zeros(b * heads * t), rig.zeros(b * heads * t * t));
    rig.record("attention backward", layers, |g| {
        let inputs = (&q, &k, &v, &probs, &dout);
        bwd::attention_bwd(g, inputs, (&dq, &dk, &dv), (&dot, &dsc), (b, t, heads, hd)).unwrap()
    });
}
