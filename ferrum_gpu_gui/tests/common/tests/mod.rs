use crate::common::signal_group;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};

mod deadline;
mod reaping;
mod signalling;

/// A throwaway process group that is always killed, so a failing assertion cannot leave
/// `sleep 60` helpers behind.
struct Group(Child);
impl Group {
    fn new(script: &str) -> Group {
        Group(
            Command::new("sh")
                .args(["-c", script])
                .process_group(0)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("start sh"),
        )
    }
    fn pgid(&self) -> u32 {
        self.0.id()
    }
}
impl Drop for Group {
    fn drop(&mut self) {
        signal_group(self.0.id(), "KILL");
        let _ = self.0.wait();
    }
}
