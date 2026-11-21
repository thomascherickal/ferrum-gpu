use super::*;

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
