use super::*;

#[test]
fn values_that_round_to_zero_are_positive_zero_like_the_cpu() {
    // Regression: the first kernel negated a float, so a small negative value came out as
    // -0.0 where the CPU (which quantises to an i8) gives +0.0. Equal as numbers, but the bit
    // patterns differ, which would break exact comparisons and checksums.
    let Some(g) = open_for_test() else { return };
    let mut blk = vec![0.0f32; 32];
    blk[0] = 127.0;
    blk[1] = -0.3; // rounds to q = 0
    blk[2] = -0.0;
    let (src, dst) = (g.upload(&blk).unwrap(), g.zeros(32).unwrap());
    q8_snap_rows(&g, &src, &dst, 32).unwrap();
    let got = g.read(&dst).unwrap();
    assert_eq!(got[1].to_bits(), 0.0f32.to_bits(), "{:e}", got[1]);
    assert_eq!(got[2].to_bits(), 0.0f32.to_bits());
}

#[test]
fn row_blocks_match_the_cpu_snap_bit_for_bit() {
    let Some(g) = open_for_test() else { return };
    for n in [32usize, 96, 4096, 64 * 100] {
        let data = nasty(n);
        let mut want = data.clone();
        snap_rows(&mut want, 32);
        let (src, dst) = (g.upload(&data).unwrap(), g.zeros(n).unwrap());
        q8_snap_rows(&g, &src, &dst, 32).unwrap();
        let got = g.read(&dst).unwrap();
        let bad = mismatches(&got, &want);
        eprintln!("rows n={n}: {bad} of {n} elements differ from the CPU snap");
        assert_eq!(bad, 0, "n={n}");
    }
}

#[test]
fn column_blocks_match_the_cpu_snap_bit_for_bit() {
    let Some(g) = open_for_test() else { return };
    for (rows_in, cols_out) in [(32usize, 1usize), (32, 7), (64, 33), (96, 128), (256, 64)] {
        let data = nasty(rows_in * cols_out);
        let mut want = data.clone();
        snap_cols(&mut want, rows_in, cols_out);
        let (src, dst) = (g.upload(&data).unwrap(), g.zeros(data.len()).unwrap());
        q8_snap_cols(&g, &src, &dst, (rows_in, cols_out)).unwrap();
        let bad = mismatches(&g.read(&dst).unwrap(), &want);
        eprintln!("cols {rows_in}x{cols_out}: {bad} elements differ");
        assert_eq!(bad, 0, "{rows_in}x{cols_out}");
    }
}

#[test]
fn ties_round_away_from_zero_and_the_scale_is_rounded_to_f16() {
    let Some(g) = open_for_test() else { return };
    // amax 127 -> d = 1 exactly; 2.5 -> 3, -2.5 -> -3 (half away from zero), 0.5 -> 1.
    let mut blk = vec![0.0f32; 32];
    blk[0] = 127.0;
    blk[1] = 2.5;
    blk[2] = -2.5;
    blk[3] = 0.5;
    blk[4] = -0.49999997; // just below the tie: must round to 0, not 1
    let (src, dst) = (g.upload(&blk).unwrap(), g.zeros(32).unwrap());
    q8_snap_rows(&g, &src, &dst, 32).unwrap();
    let got = g.read(&dst).unwrap();
    assert_eq!(&got[..5], &[127.0, 3.0, -3.0, 1.0, 0.0]);
    // A scale that f16 cannot hold exactly: amax/127 = 1/3 -> the snapped value uses f16(1/3).
    let mut blk = vec![0.0f32; 32];
    blk[0] = 127.0 / 3.0;
    blk[1] = 1.0;
    let (src, dst) = (g.upload(&blk).unwrap(), g.zeros(32).unwrap());
    q8_snap_rows(&g, &src, &dst, 32).unwrap();
    let mut want = blk.clone();
    snap_rows(&mut want, 32);
    assert_eq!(g.read(&dst).unwrap(), want);
}
