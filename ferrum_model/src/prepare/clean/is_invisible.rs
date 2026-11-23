pub(super) fn is_invisible(ch: char) -> bool {
    matches!(
        ch,
        '\u{FEFF}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}'
    )
}

/// Remove the BOM, zero-width characters, `\r`, and control characters other than `\n` and `\t`.
pub(crate) fn strip_invisible(text: &str) -> String {
    text.chars()
        .filter(|&c| !(is_invisible(c) || c == '\r' || (c.is_control() && c != '\n' && c != '\t')))
        .collect()
}

/// Unicode NFKC: ligatures, full-width letters and circled digits fold to plain forms.
pub(crate) fn nfkc(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().collect()
}

/// Curly quotes, dashes, the ellipsis and no-break spaces folded to ASCII.
pub(super) fn fold_punctuation(ch: char) -> Option<&'static str> {
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
pub(super) fn collapse_whitespace(text: &str) -> String {
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
pub(crate) fn clean(raw: &str, lowercase: bool) -> String {
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
