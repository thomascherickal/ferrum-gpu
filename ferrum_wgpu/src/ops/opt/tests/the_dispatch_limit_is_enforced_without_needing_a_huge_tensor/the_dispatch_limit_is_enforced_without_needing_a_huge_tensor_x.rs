use super::*;

#[test]
fn the_dispatch_limit_is_enforced_without_needing_a_huge_tensor() {
    assert!(check_dispatchable(1).is_ok());
    assert!(check_dispatchable(crate::runtime::GROUPS_X).is_ok());
    let e = check_dispatchable(crate::runtime::GROUPS_X + 1).unwrap_err();
    assert!(e.to_string().contains("too large"));
    // The biggest tensor that fits: partials * CHUNK elements.
    assert_eq!(
        parts_for(crate::runtime::GROUPS_X * CHUNK),
        crate::runtime::GROUPS_X
    );
}

#[test]
fn accumulate_builds_a_weighted_sum_and_first_resets_stale_contents() {
    let Some(g) = open_for_test() else { return };
    let n = 5000; // more than one 64-wide workgroup row of work
    let acc = g.upload(&vec![9.0; n]).unwrap(); // stale data that `first` must discard
    let parts: Vec<Vec<f32>> = (0..3).map(|s| rand_vec(n, 50 + s)).collect();
    let weights = [0.5f32, 0.25, 0.125];
    for (i, (p, w)) in parts.iter().zip(weights).enumerate() {
        accumulate(&g, &acc, &g.upload(p).unwrap(), w, i == 0).unwrap();
    }
    let mut want = vec![0.0f32; n];
    for (p, w) in parts.iter().zip(weights) {
        for (a, x) in want.iter_mut().zip(p) {
            *a += w * x;
        }
    }
    assert_eq!(
        g.read(&acc).unwrap(),
        want,
        "same operations in the same order"
    );
    let e = accumulate(&g, &acc, &g.zeros(3).unwrap(), 1.0, false).unwrap_err();
    assert!(e.to_string().contains("accumulate g"));
    assert_eq!(
        g.sweep_faults(|| accumulate(&g, &acc, &acc_src(&g, n), 1.0, true)),
        1
    );
}

#[test]
fn an_empty_scale_buffer_is_refused() {
    let Some(g) = open_for_test() else { return };
    let a = g.zeros(4).unwrap();
    let empty = g.zeros(0).unwrap();
    let e = adam_update(&g, (&a, &a, &a, &a), &empty, &Adam::new(0.1), true, 1).unwrap_err();
    assert!(e.to_string().contains("scale buffer is empty"));
}

#[test]
fn parts_for_counts_whole_chunks() {
    assert_eq!(parts_for(0), 1);
    assert_eq!(parts_for(1), 1);
    assert_eq!(parts_for(CHUNK), 1);
    assert_eq!(parts_for(CHUNK + 1), 2);
    assert_eq!(parts_for(10 * CHUNK), 10);
}
