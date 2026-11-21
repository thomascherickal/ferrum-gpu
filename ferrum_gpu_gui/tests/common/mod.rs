//! Launching headless Chrome against the GUI server and collecting what the page reports.
#![allow(dead_code, unused_imports)]
mod browser;
mod chrome;
mod process_group;
mod wait;

pub use browser::Browser;
pub use chrome::{chrome, has_timeout, with_deadline};
pub use process_group::{
    group_alive, group_signal_args, reap_group, signal_group, terminate_group,
};
pub use wait::wait_for;

#[cfg(test)]
#[cfg(test)]
mod tests;
