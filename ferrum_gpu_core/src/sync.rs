//! Locking that survives a panicked holder.
use std::sync::{Mutex, MutexGuard};

/// Lock `m`, recovering the data if a worker panicked while holding the lock: the state behind
/// these locks (job lists, status lines, a decoding gate) stays usable after a panic.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests;
