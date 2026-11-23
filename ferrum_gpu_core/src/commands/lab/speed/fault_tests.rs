use super::*;

/// Does a measurement fail (reporting the injected fault, not panicking) when `inject` arms a
/// fault after `k` operations? The fault is disarmed again afterwards.
fn fails_after(gpu: &Gpu, w: &Llama, inject: fn(&Gpu, usize), k: usize) -> bool {
    inject(gpu, k);
    let outcome = measure(gpu, w, None, 1);
    inject(gpu, usize::MAX);
    outcome.map(|_| false).unwrap_or_else(|e| {
        assert!(e.contains("injected fault"), "fault {k}: {e}");
        true
    })
}

/// A failed allocation stops model creation; the first readback fails while decoding and the
/// last one while prefilling (found by bisecting where the measurement starts to succeed).
#[test]
fn a_fault_in_any_stage_of_a_measurement_is_reported() {
    let Some(gpu) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let w = Llama::random(Size::Small.config(), 5, false, false);
    assert!(fails_after(&gpu, &w, Gpu::fail_allocs_after, 0));
    let read = Gpu::fail_reads_after;
    assert!(fails_after(&gpu, &w, read, 0));
    let mut hi = 1;
    while fails_after(&gpu, &w, read, hi) {
        hi *= 2;
    }
    let mut lo = hi / 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        *(if fails_after(&gpu, &w, read, mid) {
            &mut lo
        } else {
            &mut hi
        }) = mid;
    }
    assert!(fails_after(&gpu, &w, read, hi - 1) && !fails_after(&gpu, &w, read, hi));
}
