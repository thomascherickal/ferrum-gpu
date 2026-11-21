//! `train`: inputs that disagree with each other, a run that diverges, an output that cannot be
//! written.
use super::{err, ok, scratch, trained, FAST};
use ferrum_model::sha256::sha256_hex;
use ferrum_model::token_file::{write, TokenFileMeta};
mod a_diverging_run_stops_with_advice_not_nan_weights;
mod a_token_file_that_claims_another_vocabulary_than_its_tokenizer_is_refused;
