"""Reads a Markdown page into sections and picks the short labels a diagram of a section is built from."""
import re

HEADING = re.compile(r"^(#{1,3}) +(.+?)\s*$")
FENCE = re.compile(r"^```")
BOLD = re.compile(r"\*\*([^*\n]{3,40})\*\*")
BULLET = re.compile(r"^\s*(?:[-*]|\d+\.) +(.+)$")


def clean(text, width=26):
    """A node label: plain words only, cut at a word boundary."""
    plain = re.sub(r"[`*\[\]()<>|\"{}#:;]", "", re.sub(r"\]\([^)]*\)", "", text)).strip()
    words, out = plain.split(), ""
    for w in words:
        if len(out) + len(w) + 1 > width:
            break
        out = f"{out} {w}".strip()
    return out or plain[:width]


def sections(text):
    """[(title, body lines, heading line index)] for every top-level section (`##`, else `###`, else `#`)."""
    level = next((h for h in ("##", "###", "#") if re.search(rf"^{h} ", text, re.M)), "##")
    lines, found, cur, infence = text.split("\n"), [], None, False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            infence = not infence
        m = None if infence else HEADING.match(line)
        if m and m.group(1) == level:
            cur = [clean(m.group(2)), [], i]
            found.append(cur)
        elif cur is not None:
            cur[1].append(line)
    return [(t, b, i) for t, b, i in found]


def words(body):
    return len(re.sub(r"```.*?```", " ", "\n".join(body), flags=re.S).split())


def labels(body):
    """Candidate node labels, best first: sub-headings, bold terms, list items."""
    text = "\n".join(body)
    subs = [clean(m.group(2)) for l in body if (m := HEADING.match(l))]
    bold = [clean(m) for m in BOLD.findall(text)]
    items = [clean(m.group(1)) for l in body if (m := BULLET.match(l))]
    seen, out = set(), []
    for x in subs + bold + items:
        if x and x.lower() not in seen:
            seen.add(x.lower()); out.append(x)
    return out
