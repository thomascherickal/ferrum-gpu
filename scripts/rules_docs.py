"""AGENTS.md section 4: every documentation page has one Mermaid diagram per 200 words."""
import re

WORDS_PER_DIAGRAM = 200
# generator-written pages draw their own diagrams; the agent rule files are instructions, not documentation
EXEMPT = {"AGENTS.md", "CLAUDE.md", "GEMINI.md", "metrics.md", "next-steps.md", "test-time-report.md", "system-profile.md"}
FENCE = re.compile(r"^```mermaid\b", re.M)
CODE_BLOCK = re.compile(r"```.*?```", re.S)


def diagrams_missing(text):
    """How many Mermaid diagrams a page lacks (0 when it has enough)."""
    prose = CODE_BLOCK.sub(" ", text)
    return max(0, len(prose.split()) // WORDS_PER_DIAGRAM - len(FENCE.findall(text)))


def missing_diagrams(root, files):
    """{path: diagrams missing} for the Markdown pages in `files` that lack some."""
    out = {}
    for p in files:
        if p.endswith(".md") and p not in EXEMPT and not p.startswith(("docs/logs/", "fixtures/")):
            try:
                with open(f"{root}/{p}", encoding="utf-8", errors="replace") as f:
                    n = diagrams_missing(f.read())
            except OSError:
                continue
            if n:
                out[p] = n
    return out
