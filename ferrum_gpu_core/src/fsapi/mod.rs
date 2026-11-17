//! A read-only view of the machine's file system for the Studio's file pickers: list a folder,
//! summarise a file or folder, preview the start of a file. Nothing here writes, and nothing runs a
//! program. The server is loopback-only with a Host/Origin guard, and runs as the user, so this shows
//! the user their own files; the limits keep one request from reading or listing without bound.

pub mod list;
pub mod preview;
pub mod stat;

#[cfg(test)]
mod tests;

/// Most entries one listing returns.
pub const MAX_ENTRIES: usize = 2_000;
/// Most files a folder summary walks.
pub const MAX_WALK_FILES: usize = 20_000;
/// Deepest folder level a summary walks into.
pub const MAX_DEPTH: usize = 6;
/// Most bytes a preview returns.
pub const MAX_PREVIEW: usize = 64 * 1024;
/// Most bytes read to count the lines of a file.
pub const MAX_COUNT_BYTES: u64 = 16 * 1024 * 1024;
