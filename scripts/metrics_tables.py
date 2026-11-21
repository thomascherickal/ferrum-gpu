"""Part of metrics_collect.py."""
import collections
import re
import subprocess
from metrics_git import *  # noqa: F401,F403


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
