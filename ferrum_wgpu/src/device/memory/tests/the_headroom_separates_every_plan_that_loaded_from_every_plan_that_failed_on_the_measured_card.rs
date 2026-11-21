use super::*;

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
