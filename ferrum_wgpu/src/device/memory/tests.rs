use super::*;
use crate::budget::parse_nvidia_smi_free;

const GIB: u64 = 1024 * MIB;

fn facts() -> MemoryFacts {
    MemoryFacts::default()
}

/// The measured MX330 outcomes (capability.md, 2026-11-14), in MiB of memory plan: these loaded and ran,
/// these failed with "GPU out of memory". The card had 1989 MiB free of 2048.
const LOADED: [u64; 6] = [279, 557, 729, 864, 956, 1449];
const FAILED: [u64; 3] = [1721, 1925, 2454];

#[test]
fn the_headroom_separates_every_plan_that_loaded_from_every_plan_that_failed_on_the_measured_card()
{
    let f = MemoryFacts {
        tool_free_mb: Some(1989),
        tool_total_mb: Some(2048),
        ..facts()
    };
    let b = resolve_memory(None, false, &f).unwrap();
    assert_eq!(b.source, Source::ToolFree);
    for mb in LOADED {
        assert!(
            mb * MIB <= b.bytes,
            "{mb} MiB loaded, so it must fit {} bytes",
            b.bytes
        );
    }
    for mb in FAILED {
        assert!(
            mb * MIB > b.bytes,
            "{mb} MiB failed, so it must be refused at {} bytes",
            b.bytes
        );
    }
}

#[test]
fn the_override_wins_over_everything() {
    let f = MemoryFacts {
        tool_free_mb: Some(1989),
        ram_available: Some(8 * GIB),
        ..facts()
    };
    for integrated in [false, true] {
        let b = resolve_memory(Some(100), integrated, &f).unwrap();
        assert_eq!((b.bytes, b.source), (100 * MIB, Source::Override));
    }
}

#[test]
fn a_discrete_card_uses_free_memory_then_total_then_nothing() {
    let free = MemoryFacts {
        tool_free_mb: Some(10_000),
        tool_total_mb: Some(12_000),
        ..facts()
    };
    assert_eq!(
        resolve_memory(None, false, &free).unwrap().bytes,
        8_000 * MIB
    );
    let total = MemoryFacts {
        tool_total_mb: Some(10_000),
        ..facts()
    };
    let b = resolve_memory(None, false, &total).unwrap();
    assert_eq!((b.bytes, b.source), (7_000 * MIB, Source::Tool));
    assert_eq!(resolve_memory(None, false, &facts()), None);
    // RAM figures do not matter to a discrete card
    let ram_only = MemoryFacts {
        ram_available: Some(64 * GIB),
        ..facts()
    };
    assert_eq!(resolve_memory(None, false, &ram_only), None);
}

#[test]
fn an_integrated_adapter_uses_available_ram_then_half_of_total_then_nothing() {
    let avail = MemoryFacts {
        ram_available: Some(10 * GIB),
        ram_total: Some(16 * GIB),
        ..facts()
    };
    let b = resolve_memory(None, true, &avail).unwrap();
    assert_eq!((b.bytes, b.source), (7 * GIB, Source::Ram));
    let total = MemoryFacts {
        ram_total: Some(16 * GIB),
        ..facts()
    };
    let b = resolve_memory(None, true, &total).unwrap();
    assert_eq!((b.bytes, b.source), (8 * GIB, Source::HalfRam));
    assert_eq!(resolve_memory(None, true, &facts()), None);
    // a tool's VRAM figure does not matter to an integrated adapter
    let tool = MemoryFacts {
        tool_free_mb: Some(1),
        ..facts()
    };
    assert_eq!(resolve_memory(None, true, &tool), None);
}

#[test]
fn very_large_memories_do_not_overflow() {
    let f = MemoryFacts {
        ram_available: Some(u64::MAX / 2),
        ..facts()
    };
    let b = resolve_memory(None, true, &f).unwrap();
    assert!(b.bytes > (u64::MAX / 2) / 2 && b.bytes < u64::MAX / 2);
    let g = MemoryFacts {
        tool_free_mb: Some(512 * 1024),
        ..facts()
    };
    assert_eq!(
        resolve_memory(None, false, &g).unwrap().bytes,
        512 * 1024 * MIB * 8 / 10
    );
}

#[test]
fn every_budget_describes_its_source_and_says_it_is_an_estimate() {
    let f = MemoryFacts {
        tool_free_mb: Some(1989),
        tool_total_mb: Some(2048),
        ram_available: Some(10 * GIB),
        ram_total: Some(16 * GIB),
    };
    let text = |o, int| describe(&resolve_memory(o, int, &f).unwrap(), &f);
    assert!(text(Some(300), false).contains("--gpu-mem-mb"));
    let t = text(None, false);
    assert!(
        t.contains("80%") && t.contains("1989 MiB free") && t.contains("estimate"),
        "{t}"
    );
    let r = text(None, true);
    assert!(
        r.contains("70%") && r.contains("RAM") && r.contains("not measured on unified"),
        "{r}"
    );
    let only_total = MemoryFacts {
        tool_total_mb: Some(2048),
        ..MemoryFacts::default()
    };
    let t = describe(
        &resolve_memory(None, false, &only_total).unwrap(),
        &only_total,
    );
    assert!(
        t.contains("70%") && t.contains("free memory is not known"),
        "{t}"
    );
    let half = MemoryFacts {
        ram_total: Some(16 * GIB),
        ..MemoryFacts::default()
    };
    let h = describe(&resolve_memory(None, true, &half).unwrap(), &half);
    assert!(h.contains("half"), "{h}");
}

#[test]
fn nvidia_smi_free_lines_are_read_with_the_same_tolerance_as_the_total_ones() {
    let out = "NVIDIA GeForce MX330, 2048, 1989\nNVIDIA A100-SXM4-40GB, 40960, 40000\nbroken line\nName, With, Commas GPU, 8192, 100\n";
    let v = parse_nvidia_smi_free(out);
    assert_eq!(
        v[0],
        ("NVIDIA GeForce MX330".to_string(), Some(2048), Some(1989))
    );
    assert_eq!(v[1].2, Some(40000));
    assert_eq!(v.last().unwrap().0, "Name, With, Commas GPU");
    assert_eq!(v.len(), 3, "the broken line is skipped");
    let na = parse_nvidia_smi_free("Some GPU, [N/A], [N/A]\nOther, 4096, N/A\n");
    assert_eq!(na[0], ("Some GPU".to_string(), None, None));
    assert_eq!(na[1], ("Other".to_string(), Some(4096), None));
    assert!(parse_nvidia_smi_free("").is_empty());
    assert!(parse_nvidia_smi_free(", 1, 2").is_empty(), "no name");
    assert!(parse_nvidia_smi_free("x, y, z").is_empty(), "not numbers");
}
