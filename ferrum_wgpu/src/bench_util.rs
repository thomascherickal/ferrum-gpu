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

/// Tokens for a decode benchmark on a model with context `ctx`: [`DECODE_PRIME`] to fill the cache
/// first, then up to 48 to time.
pub fn decode_tokens(ctx: usize, vocab: usize) -> Vec<usize> {
    let n = 48.min(ctx.saturating_sub(8));
    (0..n + DECODE_PRIME).map(|i| (i * 7 + 3) % vocab).collect()
}

/// Tokens fed before the clock starts.
pub const DECODE_PRIME: usize = 4;

/// Feed the first [`DECODE_PRIME`] of `toks`, then time the rest: milliseconds per token.
pub fn decode_ms(toks: &[usize], mut feed: impl FnMut(usize)) -> f64 {
    let (prime, timed) = toks.split_at(DECODE_PRIME.min(toks.len()));
    prime.iter().for_each(|&t| feed(t));
    let t0 = Instant::now();
    timed.iter().for_each(|&t| feed(t));
    t0.elapsed().as_secs_f64() * 1e3 / timed.len().max(1) as f64
}

/// The adapter named by `FERRUM_GPU_ADAPTER` (else the default one), or a "nothing to benchmark"
/// line and `None`.
pub fn open_gpu() -> Option<crate::device::Gpu> {
    open_named(std::env::var("FERRUM_GPU_ADAPTER").ok().as_deref())
}

fn open_named(want: Option<&str>) -> Option<crate::device::Gpu> {
    match crate::device::Gpu::open(want) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to benchmark");
            None
        }
    }
}

#[cfg(test)]
mod tests;
