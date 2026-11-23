//! Running a program "that runs and exits" without any shell: the test binary itself. A test
//! that is also a helper does nothing unless the environment says which part to play.

use std::io;
use std::process::Command;

/// The environment variable that tells a helper test what to do.
pub const HELPER_ENV: &str = "FERRUM_TEST_HELPER";

/// The calling test binary, set to run only the test `test` (its full path, as `--exact` wants)
/// with [`HELPER_ENV`] set to `mode`.
pub fn helper_command(test: &str, mode: &str) -> io::Result<Command> {
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.args(["--exact", test, "--nocapture", "--test-threads", "1"]);
    cmd.env(HELPER_ENV, mode);
    Ok(cmd)
}

/// What this process was asked to do as a helper, if it is one.
pub fn helper_mode() -> Option<String> {
    std::env::var(HELPER_ENV).ok()
}
