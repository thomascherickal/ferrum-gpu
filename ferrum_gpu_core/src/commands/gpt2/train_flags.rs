use super::*;

/// Training-only flags of the GPT-2 path (`--arch llama` refuses them).
pub const TRAIN_FLAGS: [Flag; 22] = [
    val("--epochs"),
    val("--tokens"),
    val("--weight_decay"),
    val("--dropout"),
    val("--stride"),
    switch("--cosine"),
    val("--warmup"),
    val("--clip"),
    switch("--tie"),
    switch("--no-qat"),
    val("--tokenizer"),
    val("--checkpoint"),
    val("--checkpoint-every"),
    val("--resume"),
    val("--val"),
    val("--patience"),
    val("--max-seconds"),
    val("--metrics-csv"),
    switch("--verbose"),
    switch("--debug"),
    switch("--sample"),
    switch("--force"),
];
