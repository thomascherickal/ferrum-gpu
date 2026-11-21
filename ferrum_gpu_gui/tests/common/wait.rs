use std::time::{Duration, Instant};

/// Poll `fetch` (which returns the page's reported JSON text, if any) until it yields or `limit`
/// passes.
pub fn wait_for(limit: Duration, mut fetch: impl FnMut() -> Option<String>) -> Option<String> {
    let t0 = Instant::now();
    while t0.elapsed() < limit {
        if let Some(s) = fetch() {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}
