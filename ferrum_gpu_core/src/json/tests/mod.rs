use super::*;

fn p(s: &str) -> Result<Json, String> {
    Json::parse(s)
}

mod every_escape_a_string_may_carry_is_decoded;
mod malformed_documents_are_errors_not_panics;
