use super::*;

#[test]
fn a_lone_amd_card_falls_back_to_sysfs_but_several_cards_are_not_guessed() {
    let a = adapter("AMD Radeon RX", Kind::Discrete, "Vulkan");
    let one = Fake {
        sysfs: vec![16384],
        ..Fake::default()
    };
    let b = budget_for(None, &a, &one).0.unwrap();
    assert_eq!((b.source, b.bytes), (Source::Tool, 16384 * MIB * 70 / 100));
    let two = Fake {
        sysfs: vec![16384, 8192],
        ..Fake::default()
    };
    assert!(budget_for(None, &a, &two).0.is_none());
    // sysfs is not used for software or integrated adapters
    let soft = adapter("llvmpipe", Kind::Software, "Vulkan");
    assert_eq!(facts_for(&soft, &one).tool_total_mb, None);
}

#[test]
fn an_integrated_adapter_gets_a_share_of_the_available_ram_and_ignores_the_tool() {
    let r = Fake {
        smi: Some("NVIDIA GeForce MX330, 2048, 1989\n"),
        meminfo: Some(MEMINFO),
        ..Fake::default()
    };
    let a = adapter("Intel(R) Iris(R) Xe Graphics", Kind::Integrated, "Vulkan");
    let f = facts_for(&a, &r);
    assert_eq!((f.tool_total_mb, f.tool_free_mb), (None, None));
    let b = budget_for(None, &a, &r).0.unwrap();
    assert_eq!(b.source, Source::Ram);
    assert_eq!(b.bytes, 9313032 * 1024 * 70 / 100);
}

#[test]
fn macos_unified_memory_uses_the_total_from_sysctl_when_meminfo_is_absent() {
    let r = Fake {
        sysctl: Some("137438953472\n"),
        ..Fake::default()
    };
    let a = adapter("Apple M5 Max", Kind::Integrated, "Metal");
    let f = facts_for(&a, &r);
    assert_eq!((f.ram_total, f.ram_available), (Some(137438953472), None));
    let b = budget_for(None, &a, &r).0.unwrap();
    assert_eq!((b.source, b.bytes), (Source::HalfRam, 137438953472 / 2));
    // an unparsable sysctl answer is no answer
    let bad = Fake {
        sysctl: Some("not a number"),
        ..Fake::default()
    };
    assert_eq!(facts_for(&a, &bad).ram_total, None);
    assert!(budget_for(None, &a, &bad).0.is_none());
}

#[test]
fn the_real_system_reader_never_panics_and_answers_what_this_machine_has() {
    let s = System;
    // none of these may panic, whatever the machine; on Linux the memory report exists
    let _ = s.nvidia_smi();
    let _ = s.sysctl_memsize();
    let _ = s.sysfs_vram_mb();
    if cfg!(target_os = "linux") {
        let (total, avail) = parse_meminfo(&s.meminfo().expect("/proc/meminfo"));
        assert!(total.unwrap() > 0 && avail.unwrap() > 0);
    }
}
