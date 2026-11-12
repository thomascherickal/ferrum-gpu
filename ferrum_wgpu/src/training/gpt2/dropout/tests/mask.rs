use crate::device::open_for_test;
use crate::ops_dropout::{dropout_inplace, Place};
use ferrum_ref::dropout::Dropout;

#[test]
fn the_kernel_drops_exactly_the_units_the_cpu_mask_drops() {
    let Some(g) = open_for_test() else { return };
    // rates, seeds above 32 bits, blocks, offsets, and lengths off the workgroup size
    for (rate, seed, block, offset, n) in [
        (0.1f32, 1u64, 0u32, 0u32, 1usize),
        (0.3, 0x0123_4567_89ab_cdef, 1, 0, 100),
        (0.5, u64::MAX, 3, 7_000, 4097),
        (0.95, 42, 0, 123_456, 777),
        (0.02, 9, 5, 0, 10_000),
    ] {
        let d = Dropout::new(rate, seed).unwrap();
        let x = g.upload(&vec![1.0f32; n]).unwrap();
        dropout_inplace(&g, &x, &d, Place { block, offset }).unwrap();
        let want = d.mask_at(block as usize, offset as usize, n);
        assert_eq!(
            g.read(&x).unwrap(),
            want,
            "rate {rate} seed {seed:#x} block {block} offset {offset} n {n}"
        );
    }
}

#[test]
fn survivors_are_scaled_and_dropped_units_are_exactly_zero() {
    let Some(g) = open_for_test() else { return };
    let d = Dropout::new(0.25, 5).unwrap();
    let vals: Vec<f32> = (0..500).map(|i| i as f32 * 0.5 - 100.0).collect();
    let x = g.upload(&vals).unwrap();
    dropout_inplace(
        &g,
        &x,
        &d,
        Place {
            block: 0,
            offset: 0,
        },
    )
    .unwrap();
    let mask = d.mask(0, 500);
    let got = g.read(&x).unwrap();
    for i in 0..500 {
        let want = if mask[i] == 0.0 {
            0.0
        } else {
            vals[i] * mask[i]
        };
        assert_eq!(got[i], want, "unit {i}");
    }
    assert!(
        got.iter().filter(|&&v| v == 0.0).count() > 50,
        "some units dropped"
    );
}

#[test]
fn applying_the_mask_to_a_gradient_zeroes_the_same_units_as_the_activations() {
    // forward masks the activations, backward masks their gradient: one kernel, one mask
    let Some(g) = open_for_test() else { return };
    let d = Dropout::new(0.4, 17).unwrap();
    let place = Place {
        block: 1,
        offset: 64,
    };
    let (act, grad) = (
        g.upload(&[2.0; 300]).unwrap(),
        g.upload(&[3.0; 300]).unwrap(),
    );
    dropout_inplace(&g, &act, &d, place).unwrap();
    dropout_inplace(&g, &grad, &d, place).unwrap();
    let (a, b) = (g.read(&act).unwrap(), g.read(&grad).unwrap());
    for i in 0..300 {
        assert_eq!(a[i] == 0.0, b[i] == 0.0, "unit {i}");
    }
}

#[test]
fn unit_indices_beyond_32_bits_are_refused_by_name() {
    let Some(g) = open_for_test() else { return };
    let d = Dropout::new(0.5, 1).unwrap();
    let x = g.zeros(8).unwrap();
    let e = dropout_inplace(
        &g,
        &x,
        &d,
        Place {
            block: 0,
            offset: u32::MAX - 3,
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("32 bits"), "{e}");
}
