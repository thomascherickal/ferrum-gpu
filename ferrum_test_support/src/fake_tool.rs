//! A stand-in for an external program (`ollama`, `lms`): one description, written as a `#!/bin/sh`
//! script on Unix and as a `.cmd` batch file on Windows, with the same behaviour on both. Long
//! texts live in side files that the script prints, so no quoting rules of either shell matter.

use std::path::{Path, PathBuf};

/// What the fake program does on every call.
#[derive(Default)]
pub struct FakeTool {
    /// Append the arguments (one line per call) to this file.
    pub log: Option<PathBuf>,
    /// Also append the contents of the file named by this argument (1-based) to the log.
    pub dump_arg: Option<usize>,
    /// `(first argument, text)`: print the text when the first argument matches.
    pub replies: Vec<(String, String)>,
    /// Printed on stderr on every call.
    pub stderr: String,
    /// The exit code.
    pub exit_code: i32,
}

impl FakeTool {
    fn side(dir: &Path, name: &str, what: &str) -> PathBuf {
        dir.join(format!("{name}.{what}"))
    }

    /// The `#!/bin/sh` script, naming its side files under `dir`.
    pub fn unix_text(&self, dir: &Path, name: &str) -> String {
        let mut s = String::from("#!/bin/sh\n");
        if let Some(log) = &self.log {
            s += &format!("echo \"$@\" >> '{}'\n", log.display());
            if let Some(n) = self.dump_arg {
                s += &format!("cat \"${n}\" >> '{}'\n", log.display());
            }
        }
        for (i, (first, _)) in self.replies.iter().enumerate() {
            let file = Self::side(dir, name, &format!("reply{i}"));
            s += &format!(
                "if [ \"$1\" = '{first}' ]; then cat '{}'; fi\n",
                file.display()
            );
        }
        if !self.stderr.is_empty() {
            s += &format!("cat '{}' >&2\n", Self::side(dir, name, "err").display());
        }
        s + &format!("exit {}\n", self.exit_code)
    }

    /// The batch file, naming its side files under `dir`.
    pub fn windows_text(&self, dir: &Path, name: &str) -> String {
        let mut s = String::from("@echo off\r\n");
        if let Some(log) = &self.log {
            s += &format!(">> \"{}\" echo %*\r\n", log.display());
            if let Some(n) = self.dump_arg {
                s += &format!("type \"%~{n}\" >> \"{}\"\r\n", log.display());
            }
        }
        for (i, (first, _)) in self.replies.iter().enumerate() {
            let file = Self::side(dir, name, &format!("reply{i}"));
            s += &format!("if \"%1\"==\"{first}\" type \"{}\"\r\n", file.display());
        }
        if !self.stderr.is_empty() {
            s += &format!(
                "type \"{}\" 1>&2\r\n",
                Self::side(dir, name, "err").display()
            );
        }
        s + &format!("exit /b {}\r\n", self.exit_code)
    }

    /// Write the side files and the program for this platform into `dir`; the program's path.
    pub fn install(&self, dir: &Path, name: &str) -> PathBuf {
        for (i, (_, text)) in self.replies.iter().enumerate() {
            let file = Self::side(dir, name, &format!("reply{i}"));
            std::fs::write(file, format!("{text}\n")).expect("write a reply file");
        }
        std::fs::write(Self::side(dir, name, "err"), &self.stderr).expect("write the stderr file");
        write_program(self, dir, name)
    }
}

#[cfg(unix)]
fn write_program(tool: &FakeTool, dir: &Path, name: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, tool.unix_text(dir, name)).expect("write the script");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("make runnable");
    path
}

#[cfg(windows)]
fn write_program(tool: &FakeTool, dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(format!("{name}.cmd"));
    std::fs::write(&path, tool.windows_text(dir, name)).expect("write the batch file");
    path
}
