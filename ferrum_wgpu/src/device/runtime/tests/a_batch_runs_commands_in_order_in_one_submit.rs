use super::*;

#[test]
fn a_batch_runs_commands_in_order_in_one_submit() {
    let Some(g) = open_for_test() else { return };
    let (a, b, c) = (
        g.upload(&[1.0, 2.0, 3.0]).unwrap(),
        g.zeros(3).unwrap(),
        g.zeros(3).unwrap(),
    );
    assert!(!g.batching());
    g.begin_batch().unwrap();
    assert!(g.batching());
    // c depends on b, which depends on a: only in-order execution gives c == a.
    g.copy_f32(&a, 0, &b, 0, 3).unwrap();
    g.copy_f32(&b, 0, &c, 0, 3).unwrap();
    assert!(g.read(&c).unwrap_err().to_string().contains("batch"));
    g.end_batch(true).unwrap();
    assert!(!g.batching());
    assert_eq!(g.read(&c).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_discarded_batch_does_nothing_and_misuse_is_an_error() {
    let Some(g) = open_for_test() else { return };
    let (a, b) = (g.upload(&[5.0]).unwrap(), g.zeros(1).unwrap());
    g.begin_batch().unwrap();
    g.copy_f32(&a, 0, &b, 0, 1).unwrap();
    assert!(g
        .begin_batch()
        .unwrap_err()
        .to_string()
        .contains("already open"));
    g.end_batch(false).unwrap();
    assert_eq!(
        g.read(&b).unwrap(),
        vec![0.0],
        "discarded work must not run"
    );
    assert!(g
        .end_batch(true)
        .unwrap_err()
        .to_string()
        .contains("no batch"));
    // A dispatch inside a batch goes through the same generic path.
    let src = "struct U { v: u32, p0: u32, p1: u32, p2: u32 }
        @group(0) @binding(0) var<storage, read_write> o: array<f32>;
        @group(0) @binding(1) var<uniform> u: U;
        @compute @workgroup_size(1) fn main() { o[0] = f32(u.v); }";
    let p = g.pipeline("test-batch-dispatch", src).unwrap();
    let out = g.zeros(1).unwrap();
    g.begin_batch().unwrap();
    g.dispatch(&p, &[&out], &[7], (1, 1)).unwrap();
    g.end_batch(true).unwrap();
    assert_eq!(g.read(&out).unwrap(), vec![7.0]);
}

#[test]
fn a_bad_batched_dispatch_fails_at_end_batch_and_does_not_brick_the_device() {
    let Some(g) = open_for_test() else { return };
    // This shader declares no uniform, but dispatch always binds one: the bind group is invalid.
    let bad = "@group(0) @binding(0) var<storage, read_write> o: array<f32>; @compute @workgroup_size(1) fn main() { o[0] = 1.0; }";
    let p = g.pipeline("test-batch-bad", bad).unwrap();
    let out = g.zeros(1).unwrap();
    g.begin_batch().unwrap();
    g.dispatch(&p, &[&out], &[1], (1, 1)).unwrap(); // recording does not validate
    let e = g.end_batch(true).unwrap_err();
    assert!(e.to_string().contains("GPU error"), "{e}");
    // The error was reported once; the device keeps working.
    g.check().unwrap();
    assert_eq!(g.read(&g.upload(&[3.0]).unwrap()).unwrap(), vec![3.0]);
    // Device loss, by contrast, is sticky.
    g.destroy();
    crate::device::wait_for_loss(&g);
    assert!(g.check().unwrap_err().to_string().contains("device lost"));
    assert!(
        g.check().unwrap_err().to_string().contains("device lost"),
        "still lost on the second check"
    );
}
