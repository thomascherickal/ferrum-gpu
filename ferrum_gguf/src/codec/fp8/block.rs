use super::*;

/// Bytes of one block.
pub const BLOCK: usize = 2 + QK;

/// The two formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8 {
    /// 4 exponent bits, 3 mantissa bits, bias 7, largest value 448.
    E4M3,
    /// 5 exponent bits, 2 mantissa bits, bias 15, largest finite value 57344.
    E5M2,
}
