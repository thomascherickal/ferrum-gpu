//! Launching headless Chrome against the GUI server and collecting what the page reports.
#![allow(dead_code, unused_imports)]
mod browser;
mod chrome;
mod guard;
mod selftest;
mod wait;

pub(crate) use browser::{console_lines, Browser};
pub(crate) use chrome::chrome;
pub(crate) use guard::{with_deadline, Guarded};
pub(crate) use selftest::selftest_result;
pub(crate) use wait::wait_for;

#[cfg(test)]
mod tests;
