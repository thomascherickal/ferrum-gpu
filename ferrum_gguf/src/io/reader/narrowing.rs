use super::*;

/// Narrow a header number to the integer type the parser stores it in, or fail with `msg`
/// (on a 64-bit target the `usize` case cannot fail, so the check is generic and tested narrower).
pub fn narrow<U: TryFrom<u64>>(v: u64, msg: impl Into<String>) -> Result<U, GgufError> {
    U::try_from(v).map_err(|_| err(msg))
}
