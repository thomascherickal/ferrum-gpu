use super::*;

#[test]
fn a_request_head_that_is_not_utf8_is_refused() {
    let e = head::parse_head(b"GET /\xff HTTP/1.1\r\n\r\n").unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (400, "request head is not UTF-8")
    );
}
