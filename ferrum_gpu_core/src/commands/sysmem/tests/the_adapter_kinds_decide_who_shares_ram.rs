use super::*;

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
        ram: RAM,
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
