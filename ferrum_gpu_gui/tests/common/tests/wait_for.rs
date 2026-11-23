use crate::common::wait_for;
use std::time::Duration;

#[test]
fn waiting_returns_the_first_answer_or_gives_up_at_the_limit() {
    let mut calls = 0;
    let got = wait_for(Duration::from_secs(5), || {
        calls += 1;
        (calls == 3).then(|| "ready".to_string())
    });
    assert_eq!(got.as_deref(), Some("ready"));
    assert_eq!(wait_for(Duration::from_millis(120), || None), None);
}
