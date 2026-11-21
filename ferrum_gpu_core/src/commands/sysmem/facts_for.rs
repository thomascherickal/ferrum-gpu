use super::*;

/// What the readers say about `adapter`'s memory.
pub fn facts_for(a: &AdapterSummary, r: &dyn Readers) -> MemoryFacts {
    let mut f = MemoryFacts::default();
    if !shares_system_memory(a) {
        let smi = r.nvidia_smi().unwrap_or_default();
        if let Some((_, total, free)) = parse_nvidia_smi_free(&smi)
            .into_iter()
            .find(|(n, _, _)| names_match(n, &a.name))
        {
            f.tool_total_mb = total;
            f.tool_free_mb = free;
        } else if a.kind == Kind::Discrete {
            // sysfs cannot be matched to an adapter by name: it is used only when there is one card
            if let [only] = r.sysfs_vram_mb().as_slice() {
                f.tool_total_mb = Some(*only);
            }
        }
    }
    let (total, avail) = r.meminfo().map_or((None, None), |t| parse_meminfo(&t));
    f.ram_total = total.or_else(|| {
        r.sysctl_memsize()
            .and_then(|s| s.trim().parse::<u64>().ok())
    });
    f.ram_available = avail;
    f
}

/// The budget for `adapter` and the facts behind it: the override, else what the readers find.
pub fn budget_for(
    mem_mb: Option<u64>,
    a: &AdapterSummary,
    r: &dyn Readers,
) -> (Option<Budget>, MemoryFacts) {
    let f = facts_for(a, r);
    (resolve_memory(mem_mb, shares_system_memory(a), &f), f)
}
