//! Pure adapter selection over a list of summaries (testable without hardware).

#[cfg(test)]
mod feature_tests;
mod kind;
pub use kind::*;
mod choose;
pub use choose::*;

#[cfg(test)]
mod tests;
