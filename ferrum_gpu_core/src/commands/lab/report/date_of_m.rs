/// `YYYY-MM-DD` (UTC) for `secs` since the Unix epoch (days since 1970 turned into a calendar date).
pub fn date_of(secs: u64) -> String {
    let z = (secs / 86400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The version of crate `name` in a `Cargo.lock` text, or `unknown`.
pub fn locked_version(lock: &str, name: &str) -> String {
    let head = format!("name = \"{name}\"");
    let mut lines = lock.lines();
    while let Some(l) = lines.next() {
        if l.trim() == head {
            if let Some(v) = lines
                .next()
                .and_then(|v| v.trim().strip_prefix("version = \""))
            {
                return v.trim_end_matches('"').to_string();
            }
        }
    }
    "unknown".into()
}
