use super::*;
use ferrum_gguf::{GgufBuilder, Value};

#[test]
fn a_file_that_is_not_a_model_is_refused_on_import() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let mut other = GgufBuilder::new();
    other.meta("general.architecture", Value::Str("mamba".into()));
    for (body, why) in [
        (b"plain text, not a GGUF file".to_vec(), "magic"),
        (other.into_bytes(), "mamba"),
    ] {
        let r = call(p, "PUT", "/api/model.gguf", &body);
        assert_eq!(r.status, 400, "{}", r.text());
        assert!(r.text().contains(why), "{}", r.text());
    }
    srv.stop();
}
