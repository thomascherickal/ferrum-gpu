use super::*;

fn facts(tool: Option<(u64, Option<u64>)>, ram: Option<(u64, Option<u64>)>) -> MemoryFacts {
    MemoryFacts {
        tool_total_mb: tool.map(|t| t.0),
        tool_free_mb: tool.and_then(|t| t.1),
        ram_total: ram.map(|r| r.0 << 20),
        ram_available: ram.and_then(|r| r.1).map(|a| a << 20),
    }
}

#[test]
fn a_cards_own_memory_comes_from_the_tool_and_says_so() {
    let (src, total, free) = meter(
        &facts(Some((2048, Some(1794))), Some((16000, Some(9000)))),
        Kind::Discrete,
    )
    .unwrap();
    assert!(src.contains("nvidia-smi"));
    assert_eq!((total, free), (2048, Some(1794)));
    // a total with no free figure (sysfs only)
    let (_, total, free) = meter(&facts(Some((4096, None)), None), Kind::Discrete).unwrap();
    assert_eq!((total, free), (4096, None));
}

#[test]
fn shared_memory_reports_ram_and_a_discrete_card_without_a_tool_says_it_could_not_read_its_own() {
    let ram = Some((16000, Some(9000)));
    let (src, total, free) = meter(&facts(None, ram), Kind::Integrated).unwrap();
    assert!(src.contains("shared with the GPU") && !src.contains("could not"));
    assert_eq!((total, free), (16000, Some(9000)));
    let (src, ..) = meter(&facts(None, ram), Kind::Discrete).unwrap();
    assert!(src.contains("could not be read"));
    let (_, _, free) = meter(&facts(None, Some((8000, None))), Kind::Integrated).unwrap();
    assert_eq!(free, None);
}

#[test]
fn nothing_known_is_none_and_null() {
    assert_eq!(meter(&facts(None, None), Kind::Integrated), None);
    assert_eq!(meter_json(&facts(None, None), Kind::Discrete), Json::Null);
}

#[test]
fn the_json_carries_source_total_and_free() {
    let j = meter_json(&facts(Some((2048, Some(1794))), None), Kind::Discrete);
    assert_eq!(j.get("total_mib").and_then(Json::as_f64), Some(2048.0));
    assert_eq!(j.get("free_mib").and_then(Json::as_f64), Some(1794.0));
    assert!(j
        .get("source")
        .and_then(Json::as_str)
        .unwrap()
        .contains("nvidia-smi"));
    let none = meter_json(&facts(Some((2048, None)), None), Kind::Discrete);
    assert_eq!(none.get("free_mib"), Some(&Json::Null));
}
