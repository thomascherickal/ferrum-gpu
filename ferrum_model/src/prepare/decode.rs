use std::borrow::Cow;
use std::path::Path;

/// `raw` as text with every line ending a newline (CRLF, and the lone CR of classic Mac files).
/// Invalid UTF-8 is an error naming the byte, or with `lossy` is replaced and counted.
pub(super) fn decode<'a>(
    raw: &'a [u8],
    path: &Path,
    offset: u64,
    lossy: bool,
    repaired: &mut u64,
) -> Result<Cow<'a, str>, String> {
    let text = match std::str::from_utf8(raw) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) if lossy => {
            *repaired += 1;
            String::from_utf8_lossy(raw)
        }
        Err(e) => {
            return Err(format!(
                "{}: invalid UTF-8 at byte {} (pass --lossy to replace invalid bytes)",
                path.display(),
                offset + e.valid_up_to() as u64
            ))
        }
    };
    Ok(if text.contains('\r') {
        Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        text
    })
}
