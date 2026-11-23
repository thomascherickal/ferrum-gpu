//! The deadline and the process-group guard, from the cross-platform test-support crate (no `kill`
//! or `timeout` command, no negative pids).
pub(crate) use ferrum_test_support::{with_deadline, Guarded};
