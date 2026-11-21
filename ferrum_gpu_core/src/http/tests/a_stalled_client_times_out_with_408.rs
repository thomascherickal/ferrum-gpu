use super::*;

#[test]
fn a_stalled_client_times_out_with_408() {
    struct Stall;
    impl Read for Stall {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::WouldBlock.into())
        }
    }
    assert_eq!(read_request(&mut Stall, 10).unwrap_err().status, 408);
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::ConnectionReset.into())
        }
    }
    assert_eq!(read_request(&mut Broken, 10).unwrap_err().status, 400);
    struct StallBody(Vec<u8>);
    impl Read for StallBody {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            if self.0.is_empty() {
                return Err(std::io::ErrorKind::TimedOut.into());
            }
            b[0] = self.0.remove(0);
            Ok(1)
        }
    }
    let head = b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\nab".to_vec();
    assert_eq!(
        read_request(&mut StallBody(head), 10).unwrap_err().status,
        408
    );
}

#[test]
fn responses_serialise_with_length_and_safe_headers() {
    let mut out = Vec::new();
    let mut r = Response::new(200, "text/plain", "hi");
    r.headers.push(("X-A".into(), "b".into()));
    write_response(&mut out, &r).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(s.contains("Content-Length: 2\r\n") && s.contains("X-A: b\r\n"));
    assert!(s.contains("X-Content-Type-Options: nosniff") && s.contains("Cache-Control: no-store"));
    assert!(s.ends_with("\r\n\r\nhi"));
    for (code, text) in [
        (204, "No Content"),
        (400, "Bad Request"),
        (403, "Forbidden"),
        (404, "Not Found"),
        (405, "Method Not Allowed"),
        (408, "Request Timeout"),
        (409, "Conflict"),
        (411, "Length Required"),
        (413, "Payload Too Large"),
        (431, "Request Header Fields Too Large"),
        (500, "Internal Server Error"),
        (501, "Not Implemented"),
        (401, "Unauthorized"),
        (422, "Unprocessable Entity"),
        (503, "Service Unavailable"),
        (999, "Unknown"),
    ] {
        assert_eq!(reason(code), text);
    }
    let e = Response::error(404, "nope");
    assert_eq!(e.status, 404);
    assert_eq!(String::from_utf8(e.body).unwrap(), "{\"error\":\"nope\"}");
}
