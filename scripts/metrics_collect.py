"""Collecting the numbers behind metrics.md from the repository: files, lines, tests, dependencies,
git history. Pure functions over text and paths where possible, so they can be tested without a
repository; the few that run git take the root directory."""
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
# the rule-11 size guide: split a file that grows past about this many lines of non-test code
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


def rust_table(root, files):
    """Per crate: files, production lines, test lines, `#[test]` functions, ignored tests; plus the
    production size of every Rust file."""
    table = collections.defaultdict(lambda: dict(files=0, prod=0, test=0, tests=0, ignored=0))
    sizes = {}
    for p in files:
        if not p.endswith(".rs") or p.startswith(".claude/"):
            continue
        text = read(root, p)
        c = table[crate_of(p)]
        c["files"] += 1
        if is_test_path(p):
            c["test"] += count_lines(text)
            sizes[p] = 0
        else:
            prod, test = split_rs(text)
            c["prod"] += prod
            c["test"] += test
            sizes[p] = prod
        c["tests"] += len(TEST_ATTR.findall(text))
        c["ignored"] += len(IGNORE_ATTR.findall(text))
    return dict(table), sizes


def language_table(root, files):
    """Language -> (files, non-blank lines) over every tracked text file we recognise."""
    out = collections.defaultdict(lambda: [0, 0])
    for p in files:
        lang = LANGUAGES.get(p.rsplit(".", 1)[-1]) if "." in p else None
        if lang is None or p.startswith(".claude/"):
            continue
        out[lang][0] += 1
        out[lang][1] += count_lines(read(root, p))
    return {k: tuple(v) for k, v in out.items()}


def lock_packages(lock_text):
    """Number of packages in a Cargo.lock."""
    return len(re.findall(r"^\[\[package\]\]", lock_text, re.M))


def workspace_members(cargo_toml):
    """The member crates named in the root Cargo.toml."""
    m = re.search(r"members\s*=\s*\[(.*?)\]", cargo_toml, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def history(root):
    """Commit count, first and last commit dates, authors, branches (local) and tags."""
    count = git(root, "rev-list", "--count", "HEAD").strip() or "0"
    dates = git(root, "log", "--format=%as").split()
    authors = {a for a in git(root, "log", "--format=%an").split("\n") if a}
    branches = [b for b in git(root, "branch", "--format=%(refname:short)").split("\n") if b]
    tags = [t for t in git(root, "tag").split("\n") if t]
    return dict(
        commits=int(count),
        first=dates[-1] if dates else "-",
        last=dates[0] if dates else "-",
        authors=len(authors),
        branches=len(branches),
        tags=len(tags),
    )
