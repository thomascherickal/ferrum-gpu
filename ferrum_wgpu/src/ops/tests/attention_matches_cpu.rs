use super::*;

#[test]
fn attention_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for shape in [
        (1usize, 1usize, 1usize, 1usize),
        (1, 1, 2, 4),
        (2, 5, 2, 3),
        (1, 17, 4, 8),
        (2, 33, 3, 5),
    ] {
        let (batch, t, heads, hd) = shape;
        let n = batch * t * heads * hd;
        let (q, k, v) = (spread(n, 1, 2.0), spread(n, 2, 2.0), rand_vec(n, 3));
        let (gq, gk, gv) = (
            g.upload(&q).unwrap(),
            g.upload(&k).unwrap(),
            g.upload(&v).unwrap(),
        );
        let (gp, go) = (g.zeros(batch * heads * t * t).unwrap(), g.zeros(n).unwrap());
        attention(&g, (&gq, &gk, &gv), (&gp, &go), shape).unwrap();
        let (want_out, want_probs) = nn::attention(&q, &k, &v, shape);
        show(
            "attention probs",
            &format!("{shape:?}"),
            max_rel_err(&g.read(&gp).unwrap(), &want_probs),
            REDUCING,
        );
        show(
            "attention out",
            &format!("{shape:?}"),
            max_rel_err(&g.read(&go).unwrap(), &want_out),
            REDUCING,
        );
    }
}
