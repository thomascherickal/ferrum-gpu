use super::*;

#[test]
pub(super) fn gqa_forward_agrees_with_the_row_oracle() {
    let shape = (2, 4, 4, 2, 3);
    let (batch, t, heads, kv, hd) = shape;
    let (q, k, v) = (
        rand_vec(batch * t * heads * hd, 1),
        rand_vec(batch * t * kv * hd, 2),
        rand_vec(batch * t * kv * hd, 3),
    );
    let (out, probs) = gqa_fwd(&q, &k, &v, shape);
    let (qd, kvd) = (heads * hd, kv * hd);
    for b in 0..batch {
        for i in 0..t {
            let r = b * t + i;
            let want = attend_row(
                &q[r * qd..(r + 1) * qd],
                &k[b * t * kvd..],
                &v[b * t * kvd..],
                i + 1,
                (heads, kv, hd),
            );
            assert!(out[r * qd..(r + 1) * qd]
                .iter()
                .zip(&want)
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }
    for row in 0..batch * heads * t {
        let p = &probs[row * t..(row + 1) * t];
        assert!(
            (p.iter().sum::<f32>() - 1.0).abs() < 1e-5
                && p[row % t + 1..].iter().all(|&x| x == 0.0)
        );
    }
}

#[test]
pub(super) fn gqa_bwd_matches_finite_differences_for_every_group_ratio() {
    for (heads, kv) in [(4usize, 4usize), (4, 2), (4, 1)] {
        let shape = (2, 3, heads, kv, 2);
        let (batch, t, _, _, hd) = shape;
        let (q, k, v) = (
            rand_vec(batch * t * heads * hd, 1),
            rand_vec(batch * t * kv * hd, 2),
            rand_vec(batch * t * kv * hd, 3),
        );
        let dout = rand_vec(batch * t * heads * hd, 4);
        let (_, probs) = gqa_fwd(&q, &k, &v, shape);
        let (dq, dk, dv) = gqa_bwd((&q, &k, &v), &probs, &dout, shape);
        let f = |q: &[f32], k: &[f32], v: &[f32]| dot(&gqa_fwd(q, k, v, shape).0, &dout);
        let tag = format!("{heads}/{kv}");
        check_fd(&format!("{tag} dq"), &q, &dq, &|x| f(x, &k, &v));
        check_fd(&format!("{tag} dk"), &k, &dk, &|x| f(&q, x, &v));
        check_fd(&format!("{tag} dv"), &v, &dv, &|x| f(&q, &k, x));
    }
}
