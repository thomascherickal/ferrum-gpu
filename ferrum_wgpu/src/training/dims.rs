//! The sizes both passes of a recorded training step use.

/// `r = batch * t` rows of model width `d`, `nl` layers, feed-forward width `h`, vocabulary `v`,
/// head size `hd`, and query and key/value widths `qd` and `kvd` (both `d` for GPT-2).
pub(crate) struct Dims {
    pub(crate) b: usize,
    pub(crate) t: usize,
    pub(crate) nl: usize,
    pub(crate) r: usize,
    pub(crate) d: usize,
    pub(crate) h: usize,
    pub(crate) v: usize,
    pub(crate) hd: usize,
    pub(crate) qd: usize,
    pub(crate) kvd: usize,
}

impl Dims {
    /// The sizes of `batch` sequences of `t` tokens through `nl` layers of widths
    /// `[d, h, v, hd, qd, kvd]`.
    pub(crate) fn new((b, t): (usize, usize), nl: usize, widths: [usize; 6]) -> Dims {
        let [d, h, v, hd, qd, kvd] = widths;
        Dims {
            b,
            t,
            nl,
            r: b * t,
            d,
            h,
            v,
            hd,
            qd,
            kvd,
        }
    }
}
