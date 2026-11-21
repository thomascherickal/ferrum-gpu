//! The lab report: what a machine's adapter is, whether the GPU agrees with the CPU reference, and
//! how fast it is. Plain data with a text and a JSON rendering; nothing here touches a device, so it
//! is fully tested, and an owner of hardware we do not have can run `ferrum-gpu lab --json` and
//! send the file back.
use crate::json::Json;
mod adapterfacts;
pub use adapterfacts::*;
mod date_of;
mod rate;
mod report_json;
mod report_text;
pub use date_of::*;
pub(crate) use rate::*;

#[cfg(test)]
mod tests;
