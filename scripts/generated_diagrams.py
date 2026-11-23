"""Mermaid diagrams written by the report generators (AGENTS.md section 4); every number comes from the report."""
import re

PIE_COLORS = ["#2563eb", "#16a34a", "#d97706", "#db2777", "#7c3aed", "#0891b2", "#65a30d", "#dc2626"]
INIT = '%%{init: {"theme": "base", "themeVariables": {' + ", ".join(f'"pie{i + 1}": "{c}"' for i, c in enumerate(PIE_COLORS)) + "}}}%%"


def fence(body):
    return "```mermaid\n" + body.strip("\n") + "\n```"


def pie(title, rows):
    """A coloured pie of (label, value) rows; zero values are left out."""
    rows = [(re.sub(r'["\n]', "", str(k)), v) for k, v in rows if v]
    return fence(INIT + f"\npie showData title {title}\n" + "\n".join(f'    "{k}" : {v}' for k, v in rows)) if rows else ""


def checklist_flow(items):
    """A flow through the checklist steps (strings that start with `[x]` or `[ ]`), green when met, red when open."""
    nodes, links = [], []
    for i, item in enumerate(items):
        done = item.startswith("[x]")
        label = re.sub(r'["`]', "", re.sub(r"^\[.\] \d+\. ", "", item).split(" (")[0])[:34]
        nodes.append(f'    s{i}["{i + 1} {label}"]:::{"met" if done else "open"}')
        if i:
            links.append(f"    s{i - 1} --> s{i}")
    style = ["    classDef met fill:#dcfce7,stroke:#16a34a,color:#111", "    classDef open fill:#fee2e2,stroke:#dc2626,color:#111"]
    return fence("flowchart TD\n" + "\n".join(nodes + links + style))


def trend(title, labels, values, unit):
    """A bar chart of one number per run (newest last)."""
    if not values:
        return ""
    ys = ", ".join(str(v) for v in values)
    xs = ", ".join(f'"{x}"' for x in labels)
    return fence(f'''xychart-beta
    title "{title}"
    x-axis [{xs}]
    y-axis "{unit}" 0 --> {max(values) * 1.2 + 1:.0f}
    bar [{ys}]''')


def phase_flow(phases):
    """A flow through the phases: green when done, amber for the current one, grey for those still to come."""
    cur = next((i for i, p in enumerate(phases) if not p[1]), None)
    nodes, links = [], []
    for i, (title, done, *_rest) in enumerate(phases):
        kind = "met" if done else ("now" if i == cur else "later")
        label = " ".join(title.split(": ", 1)[-1].split()[:3])
        nodes.append(f'    p{i}["{i + 1} {label}"]:::{kind}')
        if i:
            links.append(f"    p{i - 1} --> p{i}")
    style = ["    classDef met fill:#dcfce7,stroke:#16a34a,color:#111", "    classDef now fill:#fef3c7,stroke:#d97706,color:#111",
             "    classDef later fill:#f3f4f6,stroke:#6b7280,color:#111"]
    return fence("flowchart LR\n" + "\n".join(nodes + links + style))
