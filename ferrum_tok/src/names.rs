/// Number of base tokens (one per byte value).
pub(crate) const BASE_VOCAB: usize = 256;

/// Name of the beginning-of-sequence special token.
pub const TOK_BOS: &str = "<bos>";

/// Name of the end-of-sequence special token.
pub const TOK_EOS: &str = "<|endoftext|>";

/// Name of the padding special token.
pub const TOK_PAD: &str = "<pad>";

/// Name of the unknown special token.
pub const TOK_UNK: &str = "<unk>";

/// Document separator used by prepared corpora (`train_transformer prepare`)
/// and TinyStories. [`ByteBpeTokenizer::encode_docs`] maps it to the EOS special
/// token, and [`ByteBpeTokenizer::train_with_docs`] keeps it out of the merges.
pub const DOC_SEP: &str = TOK_EOS; // "<|endoftext|>" since Phase 0-G GT
