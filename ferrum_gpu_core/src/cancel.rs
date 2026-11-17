//! Cooperative cancellation of the request running on this thread.
//!
//! The HTTP layer installs a flag for the thread that handles a request and raises it when the
//! client's connection closes; the token loops ([`crate::engine::generate_with`] and the CLI's
//! `run`) poll [`requested`] once per token and stop with [`MESSAGE`]. A request therefore frees
//! its decoder (and its in-flight slot) within one token of the client going away, instead of
//! generating to the end for nobody.
// ponytail: a thread-local, not a parameter: the loops are called through a dozen signatures
// (backends, decoders, the CLI) and every request already runs on its own thread. A model that
// fans a single request out to other threads would need the flag passed explicitly.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The error a cancelled token loop returns.
pub const MESSAGE: &str = "request cancelled: the client disconnected";

thread_local! {
    static FLAG: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Keeps `flag` installed for this thread until dropped.
pub struct Scope(());

/// Install `flag` as this thread's cancel flag.
pub fn install(flag: Arc<AtomicBool>) -> Scope {
    FLAG.with(|f| *f.borrow_mut() = Some(flag));
    Scope(())
}

impl Drop for Scope {
    fn drop(&mut self) {
        FLAG.with(|f| *f.borrow_mut() = None);
    }
}

/// Has this thread's request been cancelled? (`false` when no flag is installed.)
pub fn requested() -> bool {
    FLAG.with(|f| {
        f.borrow()
            .as_ref()
            .is_some_and(|a| a.load(Ordering::Relaxed))
    })
}

/// `Err(MESSAGE)` once cancelled, for `?` in a token loop.
pub fn check() -> Result<(), String> {
    if requested() {
        Err(MESSAGE.to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_cancelled_without_a_flag_and_the_scope_clears_it() {
        assert!(!requested() && check().is_ok());
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _s = install(flag.clone());
            assert!(!requested());
            flag.store(true, Ordering::Relaxed);
            assert!(requested());
            assert_eq!(check().unwrap_err(), MESSAGE);
        }
        assert!(!requested(), "dropping the scope uninstalls the flag");
    }

    #[test]
    fn a_flag_is_per_thread() {
        let flag = Arc::new(AtomicBool::new(true));
        let _s = install(flag);
        assert!(requested());
        assert!(!std::thread::spawn(requested).join().unwrap());
    }
}
