//! What fits: the largest model a given amount of memory can hold, and how fast it could decode.
//! These are the back-of-the-envelope formulas of `hardware-caps.md`, in Rust, so the Capable tab and the
//! document agree: `scripts/capability-estimates.py` prints the same table and a test compares them.
//! Everything here is an **estimate**: the memory formula is within -24% to +37% of the measured plans
//! (capability.md), and the speed ceiling is physics (bandwidth over weight bytes), not what ferrum-gpu reaches.
use crate::json::Json;
mod precision;
pub use precision::*;
mod usable_bytes;
pub use usable_bytes::*;
mod row_json;
pub use row_json::*;

#[cfg(test)]
mod tests;
