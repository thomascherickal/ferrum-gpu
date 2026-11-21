/// Index of block `l`'s parameter `k` (0..16) in the canonical order.
pub(super) fn pi(l: usize, k: usize) -> usize {
    2 + 16 * l + k
}
