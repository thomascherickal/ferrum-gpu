//! Turning a typed error into the plain text the commands report.
use std::fmt::Display;

/// `r` with its error as text.
pub fn text<T, E: Display>(r: Result<T, E>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_becomes_its_message_and_a_value_passes() {
        assert_eq!(
            text(Err::<(), _>(std::io::Error::other("closed"))).unwrap_err(),
            "closed"
        );
        assert_eq!(text(Ok::<_, String>(5)).unwrap(), 5);
    }
}
