use super::*;

#[test]
fn attention_bwd_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for shape in [
        (1usize, 1usize, 1usize, 1usize),
        (1, 4, 2, 3),
        (2, 5, 2, 3),
        (1, 17, 4, 8),
        (2, 33, 3, 5),
    ] {
        let (batch, t, heads, hd) = shape;
        let n = batch * t * heads * hd;
        let (q, k, v, dout) = (
            spread(n, 12, 2.0),
            spread(n, 13, 2.0),
            rand_vec(n, 14),
            rand_vec(n, 15),
        );
        let (_, probs) = nn::attention(&q, &k, &v, shape);
        let (wdq, wdk, wdv) = grad::attention_bwd(&q, &k, &v, &probs, &dout, shape);
        let up = |d: &[f32]| g.upload(d).unwrap();
        let (gq, gk, gv, gp, go) = (up(&q), up(&k), up(&v), up(&probs), up(&dout));
        let (gdq, gdk, gdv, dot, ds) = (
            g.zeros(n).unwrap(),
            g.zeros(n).unwrap(),
            g.zeros(n).unwrap(),
            g.zeros(batch * heads * t).unwrap(),
            g.zeros(batch * heads * t * t).unwrap(),
        );
        attention_bwd(
            &g,
            (&gq, &gk, &gv, &gp, &go),
            (&gdq, &gdk, &gdv),
            (&dot, &ds),
            shape,
        )
        .unwrap();
        let s = format!("{shape:?}");
        show(
            "attention_bwd dq",
            &s,
            max_rel_err(&g.read(&gdq).unwrap(), &wdq),
            REDUCING,
        );
        show(
            "attention_bwd dk",
            &s,
            max_rel_err(&g.read(&gdk).unwrap(), &wdk),
            REDUCING,
        );
        show(
            "attention_bwd dv",
            &s,
            max_rel_err(&g.read(&gdv).unwrap(), &wdv),
            REDUCING,
        );
    }
}
