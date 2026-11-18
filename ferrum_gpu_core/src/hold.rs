//! What a server binary does after it has started: wait. Forever, unless `FERRUM_SERVE_SECONDS` is set,
//! in which case it waits that long and returns so the process exits normally (which tests and CI need:
//! a process that is killed does not write its coverage data, and its exit code says it finished).

use std::time::{Duration, Instant};

/// The environment variable that bounds the wait.
pub const VAR: &str = "FERRUM_SERVE_SECONDS";

/// Wait for [`VAR`] seconds, or forever when it is not set to a number.
pub fn hold() {
    hold_for(std::env::var(VAR).ok().and_then(|s| s.parse::<f64>().ok()));
}

/// Wait `seconds` (or forever for `None`).
pub fn hold_for(seconds: Option<f64>) {
    let end = seconds.map(|s| Instant::now() + Duration::from_secs_f64(s.clamp(0.0, 1e9)));
    loop {
        let now = Instant::now();
        if end.is_some_and(|e| now >= e) {
            return;
        }
        // a bounded wait wakes at the deadline; an unbounded one wakes hourly and goes on waiting
        let wait = end.map_or(Duration::from_secs(3600), |e| {
            e.saturating_duration_since(now)
        });
        std::thread::park_timeout(wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bounded_hold_returns_after_about_its_time() {
        let t = Instant::now();
        hold_for(Some(0.1));
        let took = t.elapsed();
        assert!(
            took >= Duration::from_millis(100) && took < Duration::from_secs(5),
            "{took:?}"
        );
        // zero, and a negative or absurd number, are clamped rather than refused
        hold_for(Some(0.0));
        hold_for(Some(-5.0));
    }

    #[test]
    fn the_environment_variable_bounds_the_hold_and_anything_else_means_forever() {
        // the number is read exactly as `hold` reads it
        let read = |v: &str| v.parse::<f64>().ok();
        assert_eq!(read("2.5"), Some(2.5));
        assert_eq!(read("soon"), None);
        std::env::set_var(VAR, "0.05");
        let t = Instant::now();
        hold();
        assert!(t.elapsed() < Duration::from_secs(5));
        std::env::remove_var(VAR);
    }
}
