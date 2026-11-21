use super::*;


#[test]
fn flags_become_a_launch_or_an_error() {
    assert!(matches!(launch(&[]), Ok(Launch::Start(a)) if a.is_empty()));
    assert!(matches!(launch(&args("--device cpu")), Ok(Launch::Start(a)) if a.len() == 2));
    assert_eq!(launch(&args("--help")), Ok(Launch::Help(USAGE)));
    assert_eq!(launch(&args("-h")), Ok(Launch::Help(USAGE)));
    for bad in [
        "--device tpu",
        "--port x",
        "--nope",
        "--threads 0",
        "--device",
    ] {
        assert!(launch(&args(bad)).is_err(), "{bad}");
    }
}

#[test]
fn a_bad_device_flag_stops_before_anything_starts() {
    assert!(start_server(&args("--device tpu")).is_err());
    assert!(start_server(&args("--port 70000")).is_err());
    // asking for a GPU adapter that does not exist is an error, not a silent CPU
    assert!(start_server(&args("--device gpu --gpu-adapter no-such-adapter-zzz")).is_err());
}

#[test]
fn auto_falls_back_to_the_cpu_when_the_named_adapter_is_absent() {
    let s = start_server(&args(
        "--device auto --gpu-adapter no-such-adapter-zzz --threads 1",
    ))
    .expect("auto never fails for want of a GPU");
    let page = get(s.addr.port(), "/api/info");
    assert!(page.contains("\"active\":\"cpu\""), "{page}");
    s.stop();
}
