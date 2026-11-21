use super::*;

#[test]
fn rope_with_a_negated_sine_table_is_the_cpu_backward() {
    let Some(g) = open_for_test() else { return };
    for ty in [RopeType::Norm, RopeType::Neox] {
        let (rows, heads, hd, rd, off) = (5usize, 3usize, 8usize, 6usize, 2usize);
        let sc = RopeScaling {
            freq_factors: vec![1.0, 2.0, 4.0],
            linear_factor: 1.5,
        };
        let (c, s) = cpu::rope_tables(0, off + rows, rd, 10000.0, &sc);
        let neg: Vec<f32> = s.iter().map(|v| -v).collect();
        let dy = spread(rows * heads * hd, 7, 2.0);
        let half = rd / 2;
        let mut want = dy.clone();
        cpu::rope_bwd(
            &mut want,
            (rows, heads, hd, rd),
            (&c[off * half..], &s[off * half..]),
            ty,
        );
        let (gdy, gc, gn) = (
            g.upload(&dy).unwrap(),
            g.upload(&c).unwrap(),
            g.upload(&neg).unwrap(),
        );
        rope(&g, &gdy, (&gc, &gn), (rows, heads, hd, rd), ty, off).unwrap();
        show(
            "rope backward",
            &format!("{ty:?}"),
            max_rel_err(&g.read(&gdy).unwrap(), &want),
        );
    }
}
