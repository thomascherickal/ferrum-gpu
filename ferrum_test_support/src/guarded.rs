//! Child processes that cannot outlive their owner's intent: each runs in its own process group
//! (a job object on Windows), and ending it ends its helpers too (Chrome forks many). Nothing here
//! shells out to `kill` or `timeout`, and no group id is ever written as a negative pid.

use command_group::{CommandGroup, GroupChild};
use std::io;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

type Shared = Arc<Mutex<GroupChild>>;

/// End the group and reap its leader.
fn end(child: &Shared) {
    let mut c = child.lock().unwrap_or_else(PoisonError::into_inner);
    let _ = c.kill();
    let _ = c.wait();
}

/// A running child and its helpers, ended when this is dropped or when `max` has passed.
pub struct Guarded {
    child: Shared,
}

impl Guarded {
    /// Start `cmd` in a new group with the given output; `max` is a hard lifetime.
    pub fn spawn(
        cmd: &mut Command,
        stdout: Stdio,
        stderr: Stdio,
        max: Option<Duration>,
    ) -> io::Result<Guarded> {
        let child = cmd.stdout(stdout).stderr(stderr).group_spawn()?;
        let child = Arc::new(Mutex::new(child));
        if let Some(max) = max {
            let watched = Arc::downgrade(&child);
            std::thread::spawn(move || {
                std::thread::sleep(max);
                if let Some(child) = watched.upgrade() {
                    end(&child);
                }
            });
        }
        Ok(Guarded { child })
    }

    /// Has the leader exited?
    pub fn exited(&self) -> bool {
        let mut c = self.child.lock().unwrap_or_else(PoisonError::into_inner);
        c.try_wait().map_or(true, |status| status.is_some())
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        end(&self.child);
    }
}
