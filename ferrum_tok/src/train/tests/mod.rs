use crate::error::InferError;
use crate::test_support::{docs_corpus, naive_merges, small_tok, CORPUS};
use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP, TOK_EOS};

mod eos_reservation_edge_cases;
mod merges_compress_the_corpus;
mod no_merge_spans_whitespace;
