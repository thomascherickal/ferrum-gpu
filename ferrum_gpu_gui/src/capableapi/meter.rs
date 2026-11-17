//! The memory meter of an adapter: what can really be read, and where it came from. wgpu does not
//! report how much device memory is in use, so no "used" figure is made up: the meter shows free and
//! total from the tool or the operating system that reported them, and says which.

use crate::json::Json;
use ferrum_wgpu::memory::MemoryFacts;
use ferrum_wgpu::select::Kind;

/// `(source, total MiB, free MiB)`, or `None` when nothing is known.
pub fn meter(facts: &MemoryFacts, kind: Kind) -> Option<(&'static str, u64, Option<u64>)> {
    if let Some(total) = facts.tool_total_mb {
        return Some((
            "the card's own memory, from nvidia-smi or sysfs",
            total,
            facts.tool_free_mb,
        ));
    }
    let total = facts.ram_total? >> 20;
    let free = facts.ram_available.map(|b| b >> 20);
    Some(if kind == Kind::Discrete {
        (
            "system RAM only: this card's own memory could not be read",
            total,
            free,
        )
    } else {
        ("system RAM, shared with the GPU", total, free)
    })
}

/// The meter as JSON (`null` when nothing is known).
pub fn meter_json(facts: &MemoryFacts, kind: Kind) -> Json {
    meter(facts, kind).map_or(Json::Null, |(source, total, free)| {
        Json::obj([
            ("source", Json::Str(source.into())),
            ("total_mib", Json::Num(total as f64)),
            ("free_mib", free.map_or(Json::Null, |f| Json::Num(f as f64))),
        ])
    })
}

#[cfg(test)]
mod tests;
