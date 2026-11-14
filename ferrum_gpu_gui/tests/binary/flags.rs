use super::support::finish;

#[test]
fn help_prints_the_usage_and_exits_cleanly() {
    let (code, out, _) = finish(&["--help"]);
    assert_eq!(code, Some(0));
    assert!(out.contains("--port") && out.contains("--device"), "{out}");
}

#[test]
fn an_unknown_flag_exits_with_2_and_says_which() {
    let (code, out, err) = finish(&["--bogus"]);
    assert_eq!(code, Some(2));
    assert!(err.contains("unknown flag '--bogus'"), "{err}");
    assert!(out.is_empty(), "nothing on stdout: {out}");
}

#[test]
fn bad_values_exit_with_2_and_name_the_flag() {
    for (args, want) in [
        (&["--port", "70000"][..], "--port must be"),
        (&["--port"][..], "--port needs a value"),
        (&["--threads", "0"][..], "--threads must be at least 1"),
        (&["--device", "tpu"][..], "unknown device 'tpu'"),
    ] {
        let (code, _, err) = finish(args);
        assert_eq!(code, Some(2), "{args:?}");
        assert!(err.contains(want), "{args:?}: {err}");
    }
}

#[test]
fn a_gpu_that_is_not_there_exits_with_1_not_a_panic() {
    let (code, _, err) = finish(&["--device", "gpu", "--gpu-adapter", "no-such-adapter-xyz"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("cannot start"), "{err}");
}
