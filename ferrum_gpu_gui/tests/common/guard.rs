//! The cross-platform process-group helpers, shared with the other test crates: no `kill` or
//! `timeout` command, no negative pids; Unix process groups and Windows job objects underneath.
pub(crate) use ferrum_test_support::{with_deadline, Guarded};
