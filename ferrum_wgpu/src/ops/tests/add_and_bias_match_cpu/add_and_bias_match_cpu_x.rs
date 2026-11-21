use super::*;

#[test]
fn add_and_bias_match_cpu() {
    let Some(g) = open_for_test() else { return };
    for n in [1usize, 7, 64, 65, 1000] {
        let (a, b) = (rand_vec(n, 1), rand_vec(n, 2));
        let (ga, gb, gc) = (
            g.upload(&a).unwrap(),
            g.upload(&b).unwrap(),
            g.zeros(n).unwrap(),
        );
        add(&g, &ga, &gb, &gc).unwrap();
        show(
            "add",
            &format!("n={n}"),
            max_rel_err(&g.read(&gc).unwrap(), &nn::add(&a, &b)),
            EXACT,
        );
    }
    for (rows, cols) in [(1usize, 1usize), (3, 5), (17, 64), (2, 100)] {
        let (x, bias) = (rand_vec(rows * cols, 3), rand_vec(cols, 4));
        let (gx, gb, gy) = (
            g.upload(&x).unwrap(),
            g.upload(&bias).unwrap(),
            g.zeros(rows * cols).unwrap(),
        );
        add_bias(&g, &gx, &gb, &gy).unwrap();
        show(
            "add_bias",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&gy).unwrap(), &nn::add_bias(&x, &bias)),
            EXACT,
        );
    }
}

#[test]
fn one_d_kernels_cover_more_than_a_single_dispatch_row() {
    let Some(g) = open_for_test() else { return };
    // 4,194,240 is the old cap (65535 groups x 64). Go past it so the y dimension is used.
    let n = 4_194_240 + 1_000;
    let a: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5).collect();
    let (ga, gb, gc) = (
        g.upload(&a).unwrap(),
        g.upload(&b).unwrap(),
        g.zeros(n).unwrap(),
    );
    add(&g, &ga, &gb, &gc).unwrap();
    let got = g.read(&gc).unwrap();
    for i in [0, 1, 4_194_239, 4_194_240, 4_194_241, n - 1] {
        assert_eq!(got[i], a[i] + b[i], "element {i}");
    }
    assert_eq!(
        got,
        nn::add(&a, &b),
        "every element, including the second dispatch row"
    );
}
