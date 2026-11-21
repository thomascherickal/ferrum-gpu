/// Collapse runs of spaces and tabs to a single space.
pub fn squeeze_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_run = false;
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !in_run {
                out.push(' ');
            }
            in_run = true;
        } else {
            out.push(ch);
            in_run = false;
        }
    }
    out
}

/// The body of every Project Gutenberg book in `raw`: the text between each `*** START OF …` line
/// and the following `*** END OF` marker. `vec![raw]` when no complete START/END pair exists.
pub fn gutenberg_bodies(raw: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = 0usize;
    while let Some(s) = find_marker(&raw[rest..], "START OF") {
        let s_abs = rest + s;
        let Some(body_start) = raw[s_abs..].find('\n').map(|j| s_abs + j + 1) else {
            break;
        };
        let Some(e) = find_marker(&raw[body_start..], "END OF") else {
            break;
        };
        let e_abs = body_start + e;
        out.push(&raw[body_start..e_abs]);
        rest = raw[e_abs..].find('\n').map_or(raw.len(), |j| e_abs + j + 1);
    }
    if out.is_empty() {
        vec![raw]
    } else {
        out
    }
}

/// Offset of `*** <what>` or `***<what>`, whichever comes first (one linear pass).
pub(super) fn find_marker(s: &str, what: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(i) = s[from..].find("***") {
        let at = from + i;
        let rest = &s[at + 3..];
        if rest.starts_with(what) || rest.strip_prefix(' ').is_some_and(|r| r.starts_with(what)) {
            return Some(at);
        }
        from = at + 1;
    }
    None
}
