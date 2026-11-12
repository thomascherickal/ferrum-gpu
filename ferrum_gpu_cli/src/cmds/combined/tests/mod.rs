//! Tests for `run`: the pure command-building pieces, then the command against a real (tiny) model.

mod end_to_end;
mod pure;

fn w(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}
