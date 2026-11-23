use super::config::{parse_flags, server_config_with};
use super::*;

#[test]
fn the_token_comes_from_the_flag_then_the_environment() {
    let token = |flags: &str, env: Option<&str>| {
        let a = parse_flags(&argv(flags)).unwrap();
        server_config_with(&a, env.map(String::from)).unwrap().token
    };
    assert_eq!(token("--token flag", Some("env")).as_deref(), Some("flag"));
    assert_eq!(token("--model m", Some("env")).as_deref(), Some("env"));
    assert_eq!(token("--model m", Some("")), None, "an empty one is unset");
    assert_eq!(token("--model m", None), None);
}
