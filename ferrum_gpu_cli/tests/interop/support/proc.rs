//! Running tools: under a deadline, and servers in their own process group that are ended, with
//! their helpers, when dropped. The group handling lives in `ferrum_test_support`, shared with
//! the GUI tests that run Chrome.
use std::time::{Duration, Instant};
mod with_deadline;
pub(crate) use with_deadline::*;
mod wait_until;
pub(crate) use wait_until::*;
