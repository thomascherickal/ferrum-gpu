//! Small helpers shared by the std-only benchmarks (unit-tested here, used from `benches/`).

use std::time::{Duration, Instant};

/// GFLOP/s of an `m x k x n` multiply that took `secs` (2*m*k*n floating-point ops).
pub fn gflops(m: usize, k: usize, n: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    2.0 * m as f64 * k as f64 * n as f64 / secs / 1e9
}

/// Median of the samples (the mean of the two middle values for an even count).
pub fn median(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort();
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) / 2
    }
}

/// Run `f` once to warm up, then `reps` timed times; the median in milliseconds.
pub fn time_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut s: Vec<Duration> = (0..reps.max(1))
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    median(&mut s).as_secs_f64() * 1e3
}

/// Repetitions that keep large cases short: more for small work, one for the biggest.
pub fn reps_for(m: usize, k: usize, n: usize) -> usize {
    match 2 * m * k * n {
        0..=100_000_000 => 20,
        100_000_001..=2_000_000_000 => 7,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
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
}
