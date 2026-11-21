use super::*;

fn args(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[test]
fn flags_parse_and_unknown_or_incomplete_ones_are_refused() {
    let o = parse_args(&args(
        "--port 8080 --device gpu --gpu-adapter llvmpipe --threads 3",
    ))
    .unwrap();
    assert_eq!(
        (o.port, o.device, o.adapter.as_deref(), o.threads),
        (8080, DeviceChoice::Gpu, Some("llvmpipe"), 3)
    );
    let d = parse_args(&[]).unwrap();
    assert_eq!(
        (d.port, d.device, d.adapter, d.help),
        (0, DeviceChoice::Auto, None, false)
    );
    assert!(d.threads >= 1);
    assert!(parse_args(&args("--help")).unwrap().help);
    assert!(parse_args(&args("-h")).unwrap().help);
    for bad in [
        "--port",
        "--port x",
        "--port 70000",
        "--device",
        "--device tpu",
        "--threads 0",
        "--threads",
        "--gpu-adapter",
        "--nope",
        "stray",
    ] {
        assert!(parse_args(&args(bad)).is_err(), "{bad}");
    }
    assert!(parse_args(&args("--nope")).unwrap_err().contains("usage:"));
    for d in [DeviceChoice::Cpu, DeviceChoice::Gpu, DeviceChoice::Auto] {
        assert_eq!(DeviceChoice::parse(d.as_str()).unwrap(), d);
    }
}
