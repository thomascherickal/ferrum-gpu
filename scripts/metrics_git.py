"""Part of metrics_collect.py."""
import collections
import re
import subprocess


TEST_ATTR = re.compile(r"^\s*#\[test\]", re.M)


IGNORE_ATTR = re.compile(r"^\s*#\[ignore", re.M)


LANGUAGES = {
    "rs": "Rust", "wgsl": "WGSL shaders", "js": "JavaScript", "mjs": "JavaScript", "html": "HTML",
    "css": "CSS", "sh": "Shell", "py": "Python", "md": "Markdown", "toml": "TOML",
    "yml": "YAML", "yaml": "YAML",
}


MAX_PROD_LINES = 500


def git(root, *args):
    """Output of `git <args>` run in `root`, or "" if git fails."""
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def is_test_path(path):
    """A test file by its place or name: under a tests/ folder, or tests.rs / *_tests.rs."""
    name = path.rsplit("/", 1)[-1]
    return "/tests/" in "/" + path or name == "tests.rs" or name.endswith("_tests.rs")


def count_lines(text):
    """Non-blank lines."""
    return sum(1 for line in text.splitlines() if line.strip())


def split_rs(text):
    """(production, test) non-blank line counts of Rust source: everything from the first
    `#[cfg(test)]` line on counts as test code."""
    prod = []
    for i, line in enumerate(text.splitlines()):
        if line.strip() == "#[cfg(test)]":
            return count_lines("\n".join(prod)), count_lines("\n".join(text.splitlines()[i:]))
        prod.append(line)
    return count_lines(text), 0


def crate_of(path):
    """The top-level folder of a path, or "(root)" for a file at the top."""
    return path.split("/", 1)[0] if "/" in path else "(root)"


def read(root, path):
    try:
        with open(f"{root}/{path}", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def tracked(root):
    """Files in the index (what a commit contains)."""
    return [p for p in git(root, "ls-files").split("\n") if p]
