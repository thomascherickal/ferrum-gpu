use super::*;

fn p(s: &str) -> Result<Json, String> {
    Json::parse(s)
}

mod bytes_that_are_not_utf8_inside_a_string_are_an_error;
mod every_escape_a_string_may_carry_is_decoded;
mod malformed_documents_are_errors_not_panics;
