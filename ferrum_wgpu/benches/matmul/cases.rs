/// `(name, (m, k, n))`: the square sizes, then the reference model's matmuls.
pub(crate) fn all(sizes: &[usize]) -> Vec<(String, (usize, usize, usize))> {
    let mut cases: Vec<_> = sizes
        .iter()
        .map(|&s| (format!("{s}^3"), (s, s, s)))
        .collect();
    // Reference model (embed 96, hidden 320, vocab 1024) at 16 sequences x 128 tokens.
    for (name, k, n) in [
        ("qkv", 96, 96),
        ("ffn1", 96, 320),
        ("ffn2", 320, 96),
        ("head", 96, 1024),
    ] {
        cases.push((format!("ref-{name} (2048x{k}x{n})"), (2048, k, n)));
    }
    cases
}
