use super::*;

#[test]
fn both_kinds_of_model_start_a_server_and_say_what_they_started() {
    let d = tmp("both");
    for model in [gpt2_file(&d), llama_file(&d)] {
        let mut out = Vec::new();
        let b = boot(
            &argv(&format!("--model {model} --port 0 --device auto --token t")),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("device:") && text.contains("ferrum-gpu server: http://"),
            "{text}"
        );
        assert!(text.contains("bearer token required"), "{text}");
        assert!(b.has_token && b.addr.port() != 0);
        b.server.stop();
    }
}

#[test]
fn a_server_without_a_token_says_nothing_about_auth() {
    let d = tmp("notoken");
    let mut out = Vec::new();
    let b = boot(
        &argv(&format!("--model {} --port 0 --device cpu", gpt2_file(&d))),
        &mut out,
    )
    .unwrap();
    assert!(!String::from_utf8(out).unwrap().contains("bearer"));
    assert!(!b.has_token);
    b.server.stop();
}

#[test]
fn what_cannot_start_is_refused_with_a_reason() {
    let d = tmp("refused");
    let (g, l) = (gpt2_file(&d), llama_file(&d));
    for (args, why) in [
        (
            format!("--model {g} --device gpu --gpu-adapter no-such-adapter-xyz"),
            "adapter",
        ),
        (format!("--model {g} --port notaport"), "--port"),
        (format!("--model {g} --host 0.0.0.0 --port 0"), "loopback"),
        ("--port 0".to_string(), "--model"),
        (format!("--model {g} --threads x --port 0"), "--threads"),
        (
            format!("--model {}/missing.gguf --port 0", d.display()),
            "missing.gguf",
        ),
        (
            format!("--model {g} --device gpu --gpu-mem-mb 0 --port 0"),
            "",
        ),
        (
            format!("--model {l} --device gpu --gpu-mem-mb 0 --port 0"),
            "",
        ),
    ] {
        let e = boot(&argv(&args), &mut Vec::new())
            .err()
            .unwrap_or_else(|| panic!("{args} started"));
        assert!(e.contains(why), "{args}: {e}");
    }
}
