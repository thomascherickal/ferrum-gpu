use super::*;
use ferrum_wgpu::budget::Source;

const MIB: u64 = 1024 * 1024;

/// Fixtures instead of the machine.
#[derive(Default)]
struct Fake {
    smi: Option<&'static str>,
    meminfo: Option<&'static str>,
    sysctl: Option<&'static str>,
    sysfs: Vec<u64>,
}

impl Readers for Fake {
    fn nvidia_smi(&self) -> Option<String> {
        self.smi.map(String::from)
    }
    fn meminfo(&self) -> Option<String> {
        self.meminfo.map(String::from)
    }
    fn sysctl_memsize(&self) -> Option<String> {
        self.sysctl.map(String::from)
    }
    fn sysfs_vram_mb(&self) -> Vec<u64> {
        self.sysfs.clone()
    }
}

fn adapter(name: &str, kind: Kind, backend: &str) -> AdapterSummary {
    AdapterSummary {
        name: name.into(),
        backend: backend.into(),
        kind,
        max_buffer: 1 << 32,
        max_binding: 1 << 30,
        max_workgroup_invocations: 256,
        features: Default::default(),
    }
}

const MEMINFO: &str = "MemTotal:       16094208 kB\nMemFree:         1020000 kB\nMemAvailable:    9313032 kB\nBuffers:          100 kB\n";

#[test]
fn meminfo_gives_total_and_available_in_bytes() {
    assert_eq!(
        parse_meminfo(MEMINFO),
        (Some(16094208 * 1024), Some(9313032 * 1024))
    );
    assert_eq!(parse_meminfo("MemTotal: 100 kB\n"), (Some(102400), None));
    assert_eq!(parse_meminfo(""), (None, None));
    assert_eq!(
        parse_meminfo("MemTotal: lots kB\nMemAvailable:\n"),
        (None, None)
    );
}

#[test]
fn only_integrated_and_metal_adapters_share_system_memory() {
    assert!(shares_system_memory(&adapter(
        "Xe",
        Kind::Integrated,
        "Vulkan"
    )));
    assert!(shares_system_memory(&adapter(
        "Apple M5 Max",
        Kind::Other,
        "Metal"
    )));
    assert!(shares_system_memory(&adapter(
        "Apple GPU",
        Kind::Discrete,
        "Metal"
    )));
    assert!(!shares_system_memory(&adapter(
        "MX330",
        Kind::Discrete,
        "Vulkan"
    )));
    assert!(!shares_system_memory(&adapter(
        "llvmpipe",
        Kind::Software,
        "Vulkan"
    )));
}

#[test]
fn an_nvidia_card_gets_its_total_and_free_memory_from_the_tool_by_name() {
    let r = Fake {
        smi: Some("NVIDIA GeForce MX330, 2048, 1989\nNVIDIA A100, 40960, 40000\n"),
        meminfo: Some(MEMINFO),
        ..Fake::default()
    };
    let a = adapter("NVIDIA GeForce MX330", Kind::Discrete, "Vulkan");
    let f = facts_for(&a, &r);
    assert_eq!((f.tool_total_mb, f.tool_free_mb), (Some(2048), Some(1989)));
    let (b, _) = budget_for(None, &a, &r);
    let b = b.unwrap();
    assert_eq!(b.source, Source::ToolFree);
    assert_eq!(b.bytes, 1989 * MIB * 80 / 100);
    // the override still wins
    assert_eq!(budget_for(Some(512), &a, &r).0.unwrap().bytes, 512 * MIB);
}

#[test]
fn an_adapter_the_tool_does_not_list_gets_no_vram_figure() {
    let r = Fake {
        smi: Some("NVIDIA A100, 40960, 40000\n"),
        ..Fake::default()
    };
    let a = adapter("AMD Radeon", Kind::Discrete, "Vulkan");
    let f = facts_for(&a, &r);
    assert_eq!((f.tool_total_mb, f.tool_free_mb), (None, None));
    assert!(
        budget_for(None, &a, &r).0.is_none(),
        "unknown, so only the binding limit applies"
    );
}

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
