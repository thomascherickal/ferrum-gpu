"""Running commands for the gate, coverage and test scripts, the same way on Linux, macOS and Windows.

Only Python and the tools being run are needed: no bash, no GNU `timeout`, no `jq`, no `tee`.
"""
import os
import subprocess
import sys
import time

NOT_FOUND = 127  # what a shell reports for a program that does not exist


def environment(extra):
    """The current environment with `extra` (a dict) on top, or None to inherit it unchanged."""
    return {**os.environ, **extra} if extra else None


def run(cmd, env=None, cwd=None, stdout=None, stderr=None, timeout=None):
    """Run `cmd` (a list) and return its exit status; NOT_FOUND if the program is missing.

    With `timeout` (seconds) a command that runs longer is ended and the status is None.
    """
    try:
        done = subprocess.run(cmd, env=environment(env), cwd=cwd, stdout=stdout, stderr=stderr, timeout=timeout)
    except FileNotFoundError:
        return NOT_FOUND
    except subprocess.TimeoutExpired:
        return None
    return done.returncode


def run_logged(cmd, log, env=None, cwd=None, mode="w"):
    """Run `cmd` with stdout and stderr written to the file `log`."""
    with open(log, mode, encoding="utf-8") as f:
        return run(cmd, env, cwd, stdout=f, stderr=subprocess.STDOUT)


def help_requested(argv, doc):
    """Print `doc` and say so when `argv` asks for help, so that `--help` never starts a long run."""
    if any(a in ("-h", "--help") for a in argv):
        print(doc.strip())
        return True
    return False


def start(cmd):
    """Start `cmd` in the background; the caller waits for it."""
    return subprocess.Popen(cmd)


def run_quiet(cmd, env=None, cwd=None):
    """Run `cmd` with all of its output discarded."""
    return run(cmd, env, cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_tee(cmd, log, env=None, show=None):
    """Run `cmd`; its output (stderr merged in) goes to the file `log`, and each line that passes
    `show` (default: every line) also to the console. Returns the exit status of `cmd` itself."""
    try:
        proc = subprocess.Popen(cmd, env=environment(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
    except FileNotFoundError:
        return NOT_FOUND
    with open(log, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            if show is None or show(line):
                sys.stdout.write(line)
    return proc.wait()


def captured(cmd, cwd=None):
    """The stdout of `cmd` as text ('' if it cannot run or fails)."""
    try:
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    except FileNotFoundError:
        return ""
    return done.stdout if done.returncode == 0 else ""


def timed(fn, *args):
    """(fn(*args), seconds it took)."""
    began = time.monotonic()
    return fn(*args), time.monotonic() - began

