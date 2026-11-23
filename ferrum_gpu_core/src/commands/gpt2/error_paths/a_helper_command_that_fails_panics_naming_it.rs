use super::ok;

#[test]
#[should_panic(expected = "frobnicate now: unknown command 'frobnicate'")]
fn a_helper_command_that_fails_panics_naming_it() {
    ok("frobnicate now");
}
