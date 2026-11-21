use super::*;

/// Poll `ready` every 100 ms until it is true or `secs` pass.
pub fn wait_until(secs: u64, mut ready: impl FnMut() -> bool) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if ready() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}
