use super::rig::Rig;
use ferrum_wgpu::{ops, ops_bwd as bwd, ops_opt};

/// Cross-entropy and the embedding gradient.
pub(crate) fn loss_rows(rig: &mut Rig) {
    let (r, c, t, v) = (rig.d.rows(), rig.d.c, rig.d.t, rig.d.v);
    let (logits, grad, rl, loss) = (rig.buf(r * v), rig.zeros(r * v), rig.zeros(r), rig.zeros(1));
    let targets: Vec<usize> = (0..r).map(|i| i % v).collect();
    let tg = rig.g.upload_indices(&targets, v).unwrap();
    rig.record("cross-entropy [1024x4096]", 1, |g| {
        ops::cross_entropy(g, &logits, &tg, (&grad, &rl, &loss), (r, v)).unwrap()
    });
    let ids: Vec<usize> = (0..r).map(|i| (i * 7 + 3) % v).collect();
    let (x, dtok, dpos) = (rig.buf(r * c), rig.zeros(v * c), rig.zeros(t * c));
    rig.record("embedding gradient", 1, |g| {
        bwd::embed_bwd(g, &x, &ids, (&dtok, &dpos), (v, c, t, t)).unwrap()
    });
}

/// The optimizer over a 3.15M-element block (one transformer layer's worth, as a stand-in).
pub(crate) fn optimizer_rows(rig: &mut Rig) {
    let (n, layers) = (rig.d.h * rig.d.c, rig.d.layers);
    let (p, gr, mo, ve) = (rig.buf(n), rig.buf(n), rig.zeros(n), rig.zeros(n));
    let scale = rig.g.upload(&[1.0, 0.0]).unwrap();
    let adam = ferrum_ref::optim::Adam::new(0.001);
    rig.record("adamw update [2048x512 tensor]", 1 + layers * 2, |g| {
        ops_opt::adam_update(g, (&p, &gr, &mo, &ve), &scale, &adam, true, 1).unwrap()
    });
    let part = rig.zeros(ops_opt::parts_for(n));
    rig.record(
        "grad sum-of-squares [2048x512 tensor]",
        1 + layers * 2,
        |g| ops_opt::sumsq_parts(g, &gr, &part, 0).unwrap(),
    );
}
