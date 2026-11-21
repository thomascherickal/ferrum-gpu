//! Every WGSL kernel as source text, by `group/name`, plus the prelude the native runtime
//! prepends to each. This is what the browser runtime (Phase 5b) is served, so the page runs
//! the same source files as the native engine rather than a copy. A test keeps the list equal to
//! the files on disk and compiles each entry.

mod prelude;
pub use prelude::*;
mod sources;
pub use sources::*;
mod source;
pub use source::*;

#[cfg(test)]
mod tests;
