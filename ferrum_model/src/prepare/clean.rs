//! Per-document cleaning for `prepare`.
//!
//! Provenance: ported from ferrum-cpu `ferrum_core/src/dataset.rs` (commit 56a9d3a), reduced to the
//! options `prepare` actually uses (whitespace collapsing, punctuation folding and control-character
//! stripping always on; Gutenberg stripping and lowercasing selectable).

fn is_invisible(ch: char) -> bool {
    matches!(
        ch,
        '\u{FEFF}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}'
    )
}

/// Remove the BOM, zero-width characters, `\r`, and control characters other than `\n` and `\t`.
pub fn strip_invisible(text: &str) -> String {
    text.chars()
        .filter(|&c| !(is_invisible(c) || c == '\r' || (c.is_control() && c != '\n' && c != '\t')))
        .collect()
}

/// Unicode NFKC: ligatures, full-width letters and circled digits fold to plain forms.
pub fn nfkc(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().collect()
}

/// Curly quotes, dashes, the ellipsis and no-break spaces folded to ASCII.
fn fold_punctuation(ch: char) -> Option<&'static str> {
    Some(match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' => "'",
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => "\"",
        '\u{2013}' | '\u{2014}' | '\u{2015}' => "-",
        '\u{2026}' => "...",
        '\u{00A0}' | '\u{2007}' | '\u{202F}' => " ",
        '\u{2022}' => "*",
        _ => return None,
    })
}

/// Trim trailing line whitespace, allow at most one blank line in a row, drop leading and trailing
/// blank lines; every kept line ends with `\n`.
fn collapse_whitespace(text: &str) -> String {
    let lines: Vec<&str> = text.lines().map(str::trim_end).collect();
    let first = lines.iter().position(|l| !l.is_empty()).unwrap_or(0);
    let last = lines
        .iter()
        .rposition(|l| !l.is_empty())
        .map_or(0, |i| i + 1);
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in lines.get(first..last).unwrap_or(&[]) {
        if line.is_empty() {
            blank_run += 1;
            if blank_run >= 2 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Fold punctuation, optionally lowercase, drop control characters and collapse blank lines.
pub fn clean(raw: &str, lowercase: bool) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch == '\r' || (ch.is_control() && ch != '\n' && ch != '\t') || is_invisible(ch) {
            continue;
        }
        match fold_punctuation(ch) {
            Some(s) => out.push_str(s),
            None if lowercase => out.extend(ch.to_lowercase()),
            None => out.push(ch),
        }
    }
    collapse_whitespace(&out)
}

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
fn find_marker(s: &str, what: &str) -> Option<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_folds_and_blank_runs_collapse() {
        let got = clean(
            "\u{201C}a\u{201D} \u{2018}b\u{2019}\u{2014}c\u{2026}\r\n\n\n\nd  \n",
            false,
        );
        assert_eq!(got, "\"a\" 'b'-c...\n\nd\n");
        assert_eq!(clean("ABC\u{200B}\u{7}d", true), "abcd\n");
        assert_eq!(clean("\n\n", false), "");
    }

    #[test]
    fn invisible_characters_and_spaces() {
        assert_eq!(strip_invisible("\u{FEFF}a\u{200D}b\r\u{1}\tc\n"), "ab\tc\n");
        assert_eq!(squeeze_spaces("a  \t b\n c"), "a b\n c");
        assert_eq!(nfkc("\u{FB01}ne \u{FF21}"), "fine A");
    }

    #[test]
    fn gutenberg_keeps_every_body_and_falls_back_to_the_whole_text() {
        let raw = "junk\n*** START OF A ***\none\n*** END OF A ***\nmid\n***START OF B***\ntwo\n***END OF B***\ntail";
        assert_eq!(gutenberg_bodies(raw), vec!["one\n", "two\n"]);
        assert_eq!(gutenberg_bodies("no markers"), vec!["no markers"]);
        assert_eq!(
            gutenberg_bodies("*** START OF A ***\nopen"),
            vec!["*** START OF A ***\nopen"]
        );
        assert_eq!(gutenberg_bodies("*** START OF A"), vec!["*** START OF A"]);
        assert_eq!(
            gutenberg_bodies("**** START OF A ***\nx\n*** END OF"),
            vec!["x\n"]
        );
    }
}
