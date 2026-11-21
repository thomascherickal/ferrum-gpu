use super::*;

#[test]
fn gflops_known_value_and_zero_time() {
    assert!((gflops(1000, 1000, 1000, 1.0) - 2.0).abs() < 1e-12);
    assert_eq!(gflops(10, 10, 10, 0.0), 0.0);
}

#[test]
fn median_odd_even_empty() {
    let ms = |v: u64| Duration::from_millis(v);
    assert_eq!(median(&mut [ms(5), ms(1), ms(3)]), ms(3));
    assert_eq!(median(&mut [ms(4), ms(2), ms(8), ms(6)]), ms(5));
    assert_eq!(median(&mut []), Duration::ZERO);
}

#[test]
fn time_ms_runs_warmup_plus_reps() {
    let mut calls = 0;
    let t = time_ms(4, || calls += 1);
    assert_eq!(calls, 5);
    assert!(t >= 0.0);
    let mut once = 0;
    time_ms(0, || once += 1);
    assert_eq!(once, 2); // reps is clamped to 1
}

#[test]
fn reps_shrink_with_work() {
    assert_eq!(reps_for(64, 64, 64), 20);
    assert_eq!(reps_for(1024, 1024, 1024), 3);
    assert_eq!(reps_for(256, 256, 256), 20);
    assert_eq!(reps_for(512, 512, 512), 7);
}

#[test]
fn mib_rounds_bytes_to_whole_mebibytes() {
    assert_eq!(mib(1024 * 1024), "1");
    assert_eq!(mib(1536 * 1024), "2");
    assert_eq!(mib(0), "0");
}

#[test]
fn open_gpu_says_so_and_returns_none_for_an_adapter_that_does_not_exist() {
    assert!(open_named(Some("no-such-adapter-xyz")).is_none());
    // the environment-driven entry point opens an adapter or reports there is none, never panics
    let _ = open_gpu();
}
