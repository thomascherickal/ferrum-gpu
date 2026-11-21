use crate::device::open_for_test;

#[test]
fn round_trip_odd_lengths_and_empty() {
    let Some(g) = open_for_test() else { return };
    for n in [0usize, 1, 3, 7, 1001] {
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let b = g.upload(&data).unwrap();
        assert_eq!(b.len(), n);
        assert_eq!(b.is_empty(), n == 0);
        assert_eq!(g.read(&b).unwrap(), data, "len {n}");
    }
}

#[test]
fn shapes_are_kept_and_checked() {
    let Some(g) = open_for_test() else { return };
    let b = g
        .upload_shaped(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.len(), 6);
    assert_eq!(g.zeros(4).unwrap().shape(), &[4]);
    assert_eq!(g.zeros_shaped(&[3, 0]).unwrap().len(), 0);
    let e = g
        .upload_shaped(&[1.0, 2.0], &[2, 2])
        .expect_err("must refuse");
    assert!(e.to_string().contains("[2, 2]") && e.to_string().contains("2 were given"));
}

#[test]
fn copy_from_host_overwrites_in_place_and_checks_length() {
    let Some(g) = open_for_test() else { return };
    let b = g.upload(&[1.0, 2.0, 3.0]).unwrap();
    g.copy_from_host(&b, &[7.0, 8.0, 9.0]).unwrap();
    assert_eq!(g.read(&b).unwrap(), vec![7.0, 8.0, 9.0]);
    g.copy_from_host(&g.zeros(0).unwrap(), &[]).unwrap();
    assert!(g
        .copy_from_host(&b, &[1.0])
        .unwrap_err()
        .to_string()
        .contains("1 elements into a buffer of 3"));
    g.begin_batch().unwrap();
    assert!(g
        .copy_from_host(&b, &[1.0, 1.0, 1.0])
        .unwrap_err()
        .to_string()
        .contains("batch"));
    g.end_batch(false).unwrap();
}

#[test]
fn zeros_are_zero() {
    let Some(g) = open_for_test() else { return };
    assert_eq!(g.read(&g.zeros(5).unwrap()).unwrap(), vec![0.0; 5]);
}

#[test]
fn a_buffer_over_the_binding_limit_is_refused_by_name_of_limit() {
    let Some(g) = open_for_test() else { return };
    let too_many = (g.info().max_binding / 4 + 1) as usize;
    let e = g.zeros(too_many).expect_err("must refuse");
    assert!(e.to_string().contains("binding limit"));
}
