//! WGSL has no includes, so a kernel pulls shared text in with a marker line
//! `//! use common/<name> KEY=value ..`: the line is replaced by that fragment, recursively, with
//! each `{{KEY}}` placeholder in the fragment replaced by its value (values hold no spaces).

const MARKER: &str = "//! use ";

/// `src` with every marker line expanded from `fragments` (`(common/name, text)`). Errors name an
/// unknown fragment, a cycle, a malformed `KEY=value` pair or a placeholder left unset.
pub fn compose(src: &str, fragments: &[(&str, &str)]) -> Result<String, String> {
    expand(src, fragments, &mut Vec::new())
}

fn expand(
    src: &str,
    fragments: &[(&str, &str)],
    stack: &mut Vec<String>,
) -> Result<String, String> {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        match line.trim().strip_prefix(MARKER) {
            Some(spec) => out.push_str(&expand_use(spec, fragments, stack)?),
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

fn expand_use(
    spec: &str,
    fragments: &[(&str, &str)],
    stack: &mut Vec<String>,
) -> Result<String, String> {
    let mut words = spec.split_whitespace();
    let name = words.next().unwrap_or_default();
    let (_, text) = fragments
        .iter()
        .find(|(n, _)| *n == name)
        .ok_or_else(|| format!("unknown WGSL fragment `{name}`"))?;
    if stack.iter().any(|s| s == name) {
        return Err(format!("WGSL fragment cycle through `{name}`"));
    }
    let mut text = text.to_string();
    for pair in words {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("`{name}`: `{pair}` is not KEY=value"))?;
        text = text.replace(&format!("{{{{{key}}}}}"), value);
    }
    if text.contains("{{") {
        return Err(format!("`{name}`: a placeholder is left unset"));
    }
    stack.push(name.to_string());
    let out = expand(&text, fragments, stack);
    stack.pop();
    out
}
