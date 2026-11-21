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

/// MiB of `bytes`, rounded, for the memory tables.
pub fn mib(bytes: u64) -> String {
    format!("{:.0}", bytes as f64 / (1024.0 * 1024.0))
}

/// The adapter named by `FERRUM_GPU_ADAPTER` (else the default one), or a "nothing to benchmark"
/// line and `None`.
pub fn open_gpu() -> Option<crate::device::Gpu> {
    let want = std::env::var("FERRUM_GPU_ADAPTER").ok();
    match crate::device::Gpu::open(want.as_deref()) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to benchmark");
            None
        }
    }
}

#[cfg(test)]
mod tests;
