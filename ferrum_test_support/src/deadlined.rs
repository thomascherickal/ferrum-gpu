//! A command that must finish within a time limit, or it and its helpers are killed.

use crate::guarded::Guarded;
use command_group::CommandGroup;
use std::ffi::OsStr;
use std::io::{self, Read};
use std::process::{Command, Output, Stdio};
use std::thread::{sleep, JoinHandle};
use std::time::{Duration, Instant};

/// `program` that gets `secs` seconds; the stand-in for coreutils `timeout`, on every platform.
pub fn with_deadline(program: impl AsRef<OsStr>, secs: u64) -> Deadlined {
    Deadlined::new(Command::new(program), secs)
}

/// A command and its time limit, built like a [`Command`].
pub struct Deadlined {
    cmd: Command,
    limit: Duration,
}

fn drain(source: Option<impl Read + Send + 'static>) -> Option<JoinHandle<Vec<u8>>> {
    source.map(|mut r| {
        std::thread::spawn(move || {
            let mut bytes = vec![];
            let _ = r.read_to_end(&mut bytes);
            bytes
        })
    })
}

impl Deadlined {
    /// `cmd` with `secs` seconds to finish.
    pub fn new(cmd: Command, secs: u64) -> Deadlined {
        Deadlined {
            cmd,
            limit: Duration::from_secs(secs),
        }
    }

    /// Add an argument.
    pub fn arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
        self.cmd.arg(arg);
        self
    }

    /// Add arguments.
    pub fn args<S: AsRef<OsStr>>(&mut self, args: impl IntoIterator<Item = S>) -> &mut Self {
        self.cmd.args(args);
        self
    }

    /// Set an environment variable.
    pub fn env(&mut self, key: impl AsRef<OsStr>, val: impl AsRef<OsStr>) -> &mut Self {
        self.cmd.env(key, val);
        self
    }

    /// Set the standard input.
    pub fn stdin(&mut self, stdin: Stdio) -> &mut Self {
        self.cmd.stdin(stdin);
        self
    }

    /// Run to the end and collect the output; past the limit the whole group is killed and the
    /// status says it did not succeed.
    pub fn output(&mut self) -> io::Result<Output> {
        let mut child = self
            .cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .group_spawn()?;
        let (out, err) = (
            drain(child.inner().stdout.take()),
            drain(child.inner().stderr.take()),
        );
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= self.limit {
                child.kill()?;
                break child.wait()?;
            }
            sleep(Duration::from_millis(20));
        };
        Ok(Output {
            status,
            stdout: out.and_then(|h| h.join().ok()).unwrap_or_default(),
            stderr: err.and_then(|h| h.join().ok()).unwrap_or_default(),
        })
    }

    /// Start it in the background with `stdout` and `stderr`, ended on drop or after the limit.
    pub fn guard(&mut self, stdout: Stdio, stderr: Stdio) -> io::Result<Guarded> {
        Guarded::spawn(&mut self.cmd, stdout, stderr, Some(self.limit))
    }
}
