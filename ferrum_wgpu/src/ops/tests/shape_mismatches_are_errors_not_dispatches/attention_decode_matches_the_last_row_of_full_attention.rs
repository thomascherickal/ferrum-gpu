use super::*;

#[test]
fn attention_decode_matches_the_last_row_of_full_attention() {
    let Some(g) = open_for_test() else { return };
    let (heads, hd, t, ctx) = (3usize, 5usize, 9usize, 12usize);
    let c = heads * hd;
    let (q, k, v) = (
        spread(t * c, 31, 2.0),
        spread(t * c, 32, 2.0),
        rand_vec(t * c, 33),
    );
    let (want_out, _) = nn::attention(&q, &k, &v, (1, t, heads, hd));
    for len in [1usize, 4, t] {
        // Cache holds positions 0..len; the query is position len-1.
        let mut kc = vec![0.0; ctx * c];
        let mut vc = vec![0.0; ctx * c];
        kc[..len * c].copy_from_slice(&k[..len * c]);
        vc[..len * c].copy_from_slice(&v[..len * c]);
        let (gq, gk, gv) = (
            g.upload(&q[(len - 1) * c..len * c]).unwrap(),
            g.upload(&kc).unwrap(),
            g.upload(&vc).unwrap(),
        );
        let (gp, go) = (g.zeros(heads * ctx).unwrap(), g.zeros(c).unwrap());
        attention_decode(&g, (&gq, &gk, &gv), (&gp, &go), (heads, hd, len, ctx)).unwrap();
        let want = &want_out[(len - 1) * c..len * c];
        show(
            "attention_decode",
            &format!("len={len}"),
            max_rel_err(&g.read(&go).unwrap(), want),
            REDUCING,
        );
    }
    let z = g.zeros(c).unwrap();
    let big = g.zeros(ctx * c).unwrap();
    let pr = g.zeros(heads * ctx).unwrap();
    assert!(attention_decode(&g, (&z, &big, &big), (&pr, &z), (heads, hd, 0, ctx)).is_err());
    assert!(attention_decode(&g, (&z, &big, &big), (&pr, &z), (heads, hd, ctx + 1, ctx)).is_err());
}

#[test]
fn embed_at_uses_the_absolute_position() {
    let Some(g) = open_for_test() else { return };
    let (vocab, c, t) = (6usize, 4usize, 5usize);
    let (tok, pos) = (rand_vec(vocab * c, 41), rand_vec(t * c, 42));
    for p in [0usize, 3, 4] {
        let gi = g.upload_indices(&[2], vocab).unwrap();
        let (gt, gp, go) = (
            g.upload(&tok).unwrap(),
            g.upload(&pos).unwrap(),
            g.zeros(c).unwrap(),
        );
        embed_at(&g, (&gt, &gp, &gi), &go, (1, c, t), p).unwrap();
        let want: Vec<f32> = (0..c).map(|d| tok[2 * c + d] + pos[p * c + d]).collect();
        show(
            "embed_at",
            &format!("pos={p}"),
            max_rel_err(&g.read(&go).unwrap(), &want),
            EXACT,
        );
    }
}

#[test]
fn repeat_runs_are_bit_identical() {
    let Some(g) = open_for_test() else { return };
    let (rows, vocab) = (20usize, 50usize);
    let logits = spread(rows * vocab, 21, 5.0);
    let targets: Vec<usize> = (0..rows).map(|i| i % vocab).collect();
    let run = || {
        let (gl, gt) = (
            g.upload(&logits).unwrap(),
            g.upload_indices(&targets, vocab).unwrap(),
        );
        let (gg, gr, gloss) = (
            g.zeros(rows * vocab).unwrap(),
            g.zeros(rows).unwrap(),
            g.zeros(1).unwrap(),
        );
        cross_entropy(&g, &gl, &gt, (&gg, &gr, &gloss), (rows, vocab)).unwrap();
        (g.read(&gg).unwrap(), g.read(&gloss).unwrap())
    };
    assert_eq!(run(), run());
}
