use super::*;

#[test]
fn device_choices_parse_and_print_the_same_words() {
    for (s, c) in [
        ("cpu", DeviceChoice::Cpu),
        ("gpu", DeviceChoice::Gpu),
        ("auto", DeviceChoice::Auto),
    ] {
        assert_eq!(DeviceChoice::parse(s), Ok(c));
        assert_eq!(c.as_str(), s);
    }
    let e = DeviceChoice::parse("tpu").unwrap_err();
    assert!(e.contains("'tpu'") && e.contains("cpu, gpu, auto"), "{e}");
    assert!(DeviceChoice::parse("").is_err());
    assert!(DeviceChoice::parse("CPU").is_err(), "the spelling is exact");
}

#[test]
fn a_cpu_session_has_a_cpu_backend_and_no_note() {
    let s = Session::new(DeviceChoice::Cpu, None, 2).unwrap();
    assert_eq!(s.backend.kind(), "cpu");
    assert_eq!(s.requested, DeviceChoice::Cpu);
    assert_eq!(s.fallback_note, None);
}

#[test]
fn auto_falls_back_to_the_cpu_with_a_note_when_the_named_adapter_is_absent() {
    let s = Session::new(DeviceChoice::Auto, Some("no-such-adapter-zzz"), 1).unwrap();
    assert_eq!(s.backend.kind(), "cpu");
    let note = s.fallback_note.expect("a note");
    assert!(
        note.contains("no usable GPU") && note.contains("using the CPU"),
        "{note}"
    );
    assert_eq!(s.requested, DeviceChoice::Auto);
}

#[test]
fn gpu_with_a_missing_adapter_is_an_error_not_a_silent_cpu() {
    let e = Session::new(DeviceChoice::Gpu, Some("no-such-adapter-zzz"), 1)
        .err()
        .expect("an error");
    assert!(!e.is_empty());
}

#[test]
fn with_backend_keeps_what_it_is_given() {
    let b: Arc<dyn Backend> = Arc::new(CpuBackend::new(1));
    let s = Session::with_backend(b, DeviceChoice::Gpu, Some("note".into()), vec![]);
    assert_eq!(s.requested, DeviceChoice::Gpu);
    assert_eq!(s.fallback_note.as_deref(), Some("note"));
    assert!(s.adapters.is_empty());
}
