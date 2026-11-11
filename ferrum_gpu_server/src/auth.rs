//! Bearer-token gate: every route except `GET /health` needs `Authorization: Bearer <token>`.

use ferrum_gpu_gui::http::{Request, Response};

/// Compare without an early exit, so the time does not depend on how much of the token matched.
fn same(a: &[u8], b: &[u8]) -> bool {
    let n = a.len().max(b.len());
    let diff = (0..n).fold(a.len() ^ b.len(), |acc, i| {
        acc | usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0))
    });
    diff == 0
}

/// `Some(401)` when `req` needs the token and does not carry it.
pub fn check(req: &Request, token: &str) -> Option<Response> {
    if req.method == "GET" && req.path == "/health" {
        return None;
    }
    let ok = req
        .header("authorization")
        .and_then(|v| v.split_once(' '))
        .is_some_and(|(scheme, t)| {
            scheme.eq_ignore_ascii_case("bearer") && same(t.as_bytes(), token.as_bytes())
        });
    if ok {
        return None;
    }
    let mut r = Response::error(401, "missing or invalid bearer token");
    r.headers.push(("WWW-Authenticate".into(), "Bearer".into()));
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, path: &str, auth: Option<&str>) -> Request {
        Request {
            method: method.into(),
            path: path.into(),
            query: vec![],
            headers: auth
                .map(|a| vec![("authorization".to_string(), a.to_string())])
                .unwrap_or_default(),
            body: vec![],
        }
    }

    #[test]
    fn health_is_open_everything_else_needs_the_token() {
        assert!(check(&req("GET", "/health", None), "s3cret").is_none());
        for r in [
            req("POST", "/health", None),
            req("GET", "/v1/info", None),
            req("POST", "/v1/generate", Some("Bearer wrong")),
            req("POST", "/v1/generate", Some("Basic s3cret")),
            req("POST", "/v1/generate", Some("s3cret")),
        ] {
            let denied = check(&r, "s3cret").expect("must be refused");
            assert_eq!(denied.status, 401);
            assert!(denied
                .headers
                .iter()
                .any(|(k, v)| k == "WWW-Authenticate" && v == "Bearer"));
        }
        assert!(check(
            &req("POST", "/v1/generate", Some("Bearer s3cret")),
            "s3cret"
        )
        .is_none());
        assert!(check(&req("GET", "/v1/info", Some("bearer s3cret")), "s3cret").is_none());
    }

    #[test]
    fn comparison_covers_length_and_content() {
        assert!(same(b"abc", b"abc") && same(b"", b""));
        assert!(!same(b"abc", b"abd") && !same(b"abc", b"ab") && !same(b"ab", b"abc"));
    }
}
