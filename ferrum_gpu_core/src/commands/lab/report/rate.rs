/// Tokens per second written without false precision: `1234`, `56`, `7.8`.
pub(crate) fn rate(v: f64) -> String {
    if !v.is_finite() || v < 0.0 {
        "n/a".into()
    } else if v >= 10.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}
