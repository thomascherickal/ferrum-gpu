"""The decisions of time_limit.py, apart from running anything: which executables, which limit,
what a result means, how the summary reads."""
import json
import os

NOT_FOUND = 127


def parse_artifacts(lines):
    """Sorted unique (manifest dir, executable) of every test executable in `cargo ... --message-format=json`."""
    found = set()
    for raw in lines:
        try:
            m = json.loads(raw)
        except ValueError:
            continue
        if m.get("reason") == "compiler-artifact" and m.get("profile", {}).get("test") and m.get("executable"):
            found.add((os.path.dirname(m["manifest_path"]), m["executable"]))
    return sorted(found)


def parse_limit(text):
    """A limit in whole seconds, or None when `text` is not one."""
    return int(text) if text.isascii() and text.isdigit() else None


def limit_for(executable, general, interop):
    """The interoperability binaries get their own limit; 0 means no limit."""
    return interop if os.path.basename(executable).startswith("interop-") else general


def classify(returncode, limit):
    """pass, hang (ended at its limit; `returncode` None) or fail."""
    if returncode == 0:
        return "pass"
    return "hang" if limit and returncode is None else "fail"


def listed_tests(listing):
    """The test names in the output of `--list --format terse`."""
    return [l[: -len(": test")] for l in listing.splitlines() if l.endswith(": test")]


def summary(times, general, interop, counts, failed, hung):
    """The closing lines: the eight slowest tests, the result line and the names that went wrong."""
    slowest = sorted(times, key=lambda t: -t[0])[:8]
    out = ["", f"== slowest tests (limits: ordinary {general}s [0 = none], interop {interop}s) =="]
    out += [f"{secs:.9f} {binary} {name}" for secs, binary, name in slowest]
    out += ["", "RESULT: {pass} passed, {fail} failed, {hang} over their limit".format(**counts)]
    if failed:
        out.append("FAILED:" + "".join(f"\n  {n}" for n in failed))
    if hung:
        out.append("OVER THE LIMIT:" + "".join(f"\n  {n}" for n in hung))
    return out
