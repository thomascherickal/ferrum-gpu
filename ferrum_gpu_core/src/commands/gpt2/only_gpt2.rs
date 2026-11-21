use super::*;

/// The first flag in `a` that only the GPT-2 path understands, if any.
pub fn only_gpt2(a: &Args) -> Option<&'static str> {
    TRAIN_FLAGS
        .iter()
        .chain(SIZE_FLAGS.iter())
        .map(|f| f.name)
        .filter(|n| *n != "--preset" && *n != "--context")
        .find(|n| a.has(n))
}

/// Generation flags: ours and ferrum-cpu's spellings.
pub const GEN_FLAGS: [Flag; 9] = [
    val("--temp"),
    val("--gen-seed"),
    val("--max-tokens"),
    val("--chars"),
    val("--top-p"),
    val("--repetition-penalty"),
    many("--stop"),
    switch("--stream"),
    val("--ids"),
];
