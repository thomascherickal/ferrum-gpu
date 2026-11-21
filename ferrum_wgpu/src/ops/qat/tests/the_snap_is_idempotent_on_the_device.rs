use super::*;

#[test]
fn the_snap_is_idempotent_on_the_device() {
    let Some(g) = open_for_test() else { return };
    let data = rand_vec(64 * 4, 12);
    let (a, b, c) = (
        g.upload(&data).unwrap(),
        g.zeros(256).unwrap(),
        g.zeros(256).unwrap(),
    );
    q8_snap_cols(&g, &a, &b, (64, 4)).unwrap();
    q8_snap_cols(&g, &b, &c, (64, 4)).unwrap();
    assert_eq!(g.read(&b).unwrap(), g.read(&c).unwrap());
}

#[test]
fn ineligible_shapes_are_copied_unchanged() {
    let Some(g) = open_for_test() else { return };
    let data = rand_vec(8 * 5, 13);
    let (src, dst) = (g.upload(&data).unwrap(), g.zeros(40).unwrap());
    q8_snap_cols(&g, &src, &dst, (8, 5)).unwrap();
    assert_eq!(g.read(&dst).unwrap(), data);
    let (src, dst) = (g.upload(&data).unwrap(), g.zeros(40).unwrap());
    q8_snap_rows(&g, &src, &dst, 8).unwrap();
    assert_eq!(g.read(&dst).unwrap(), data);
    // Eligible row length but a tensor that is not a whole number of blocks.
    let odd = rand_vec(40, 14);
    let (src, dst) = (g.upload(&odd).unwrap(), g.zeros(40).unwrap());
    q8_snap_rows(&g, &src, &dst, 32).unwrap();
    assert_eq!(g.read(&dst).unwrap(), odd);
}

#[test]
fn shape_errors_and_faults_are_reported() {
    let Some(g) = open_for_test() else { return };
    let (a, b) = (g.zeros(64).unwrap(), g.zeros(32).unwrap());
    assert!(q8_snap_rows(&g, &a, &b, 32)
        .unwrap_err()
        .to_string()
        .contains("destination"));
    assert!(q8_snap_cols(&g, &a, &b, (32, 2)).is_err());
    let c = g.zeros(64).unwrap();
    assert!(q8_snap_cols(&g, &a, &c, (32, 3))
        .unwrap_err()
        .to_string()
        .contains("is not 32 x 3"));
    assert_eq!(g.sweep_faults(|| q8_snap_cols(&g, &a, &c, (32, 2))), 1);
    assert_eq!(g.sweep_faults(|| q8_snap_rows(&g, &a, &c, 32)), 1);
    assert_eq!(
        g.sweep_faults(|| q8_snap_rows(&g, &a, &c, 8)),
        1,
        "the copy path is a command too"
    );
}
