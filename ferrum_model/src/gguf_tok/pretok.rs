/// Dependency-free approximation of GPT-2 pre-tokenization: split into runs of
/// letters / digits / symbols, with a single leading space attached to the
/// following run (so the space becomes the BPE `Ġ` marker). Whitespace runs are
/// emitted as their own chunks.
pub(super) fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        if chars[i] == ' ' {
            i += 1; // attach one leading space to the following run
        }
        if i >= chars.len() {
            chunks.push(chars[start..].iter().collect());
            break;
        }
        let c = chars[i];
        if c.is_alphabetic() {
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
        } else if c.is_numeric() {
            while i < chars.len() && chars[i].is_numeric() {
                i += 1;
            }
        } else if c.is_whitespace() {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
        } else {
            while i < chars.len() && !chars[i].is_alphanumeric() && !chars[i].is_whitespace() {
                i += 1;
            }
        }
        chunks.push(chars[start..i].iter().collect());
    }
    chunks
}
