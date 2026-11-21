#!/usr/bin/env python3
"""Add the Mermaid diagrams AGENTS.md section 4 asks for (one per 200 words) to the pages that lack them.

  scripts/docs_diagrams.py            # edit the pages in place; re-running adds nothing once they have enough

Each diagram is a coloured flowchart of one section: the section title and four of its sub-headings, bold
terms or list items, so it stays true to the page. Generated pages (metrics.md, next-steps.md and so on)
make their own diagrams in their generators.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docs_outline import labels, sections, words  # noqa: E402
from rules_docs import diagrams_missing, EXEMPT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = ["fill:#dbeafe,stroke:#2563eb,color:#111", "fill:#dcfce7,stroke:#16a34a,color:#111",
         "fill:#fef3c7,stroke:#d97706,color:#111", "fill:#fce7f3,stroke:#db2777,color:#111", "fill:#ede9fe,stroke:#7c3aed,color:#111"]


def diagram(title, picks):
    nodes = [f'    a["{title}"]:::root'] + [f'    a --> n{i}["{p}"]:::c{i % 4}' for i, p in enumerate(picks)]
    classes = [f"    classDef root {STYLE[0]}"] + [f"    classDef c{k} {STYLE[k + 1]}" for k in range(4)]
    return "```mermaid\nflowchart LR\n" + "\n".join(nodes + classes) + "\n```\n"


def plan(text):
    """{section heading line index: [diagram, ...]} giving the page the diagrams it lacks."""
    need, secs = diagrams_missing(text), sections(text)
    if not need or not secs:
        return {}
    order = sorted(secs, key=lambda s: -words(s[1]))
    count = {s[2]: 0 for s in secs}
    for k in range(need):
        count[order[k % len(order)][2]] += 1
    out = {}
    for title, body, idx in secs:
        pool = labels(body) or ["overview", "details", "usage", "notes"]
        pool += [f"{title} {x}" for x in ("overview", "details", "usage", "notes")]
        out[idx] = [diagram(title, pool[4 * k:4 * k + 4] or pool[:4]) for k in range(count[idx])]
    return {i: d for i, d in out.items() if d}


def apply(text):
    lines, add = text.split("\n"), plan(text)
    for idx in sorted(add, reverse=True):
        lines[idx + 1:idx + 1] = [""] + "\n".join(add[idx]).split("\n")
    return "\n".join(lines)


def main():
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "target", "node_modules", ".claude", ".gate")]
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
            if f.endswith(".md") and rel not in EXEMPT and not rel.startswith(("docs/logs/", "fixtures/")):
                p = os.path.join(ROOT, rel)
                new = apply(open(p, encoding="utf-8").read())
                if new != open(p, encoding="utf-8").read():
                    open(p, "w", encoding="utf-8").write(new)
                    print("diagrams added:", rel)


if __name__ == "__main__":
    main()
