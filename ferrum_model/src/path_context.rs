//! The `path: reason` error text the commands give for a failed file operation.
use std::fmt::Display;

/// Prefix the error of `r` with the path it happened on.
pub fn at<T, E: Display>(r: Result<T, E>, path: impl Display) -> Result<T, String> {
    r.map_err(|e| format!("{path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_is_prefixed_with_its_path_and_a_value_passes() {
        let failed = Err::<(), _>(std::io::Error::other("no such file"));
        assert_eq!(
            at(failed, "/gone/file").unwrap_err(),
            "/gone/file: no such file"
        );
        assert_eq!(at(Ok::<_, String>(7), "p").unwrap(), 7);
    }
}
