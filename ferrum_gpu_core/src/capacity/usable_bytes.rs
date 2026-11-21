use super::*;

/// Usable bytes of `memory_gb` gigabytes (10^9 bytes) of discrete or unified memory.
pub fn usable_bytes(memory_gb: f64, unified: bool) -> f64 {
    memory_gb
        * 1e9
        * if unified {
            USABLE_UNIFIED
        } else {
            USABLE_DISCRETE
        }
}

/// The memory-bandwidth ceiling on decode speed: every token reads every weight once.
pub fn decode_ceiling_tps(bandwidth_gb_s: f64, params: f64, p: Precision) -> f64 {
    let bytes = params * p.bytes_per_weight();
    if bytes <= 0.0 {
        return 0.0;
    }
    bandwidth_gb_s * 1e9 / bytes
}

/// A parameter count as the documents write it: `1.5B` from one billion up, `250M` below.
pub fn format_params(params: f64) -> String {
    let b = params / 1e9;
    if b >= 1.0 {
        format!("{b:.1}B")
    } else {
        format!("{:.0}M", b * 1000.0)
    }
}

/// One memory size's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// Memory in gigabytes.
    pub memory_gb: f64,
    /// Usable bytes after the share for the system.
    pub usable: f64,
    /// Largest inference model per precision (Q4_0, Q8_0, f32), in parameters.
    pub infer: [f64; 3],
    /// Largest full fine-tune (f32 AdamW), in parameters.
    pub train: f64,
}

/// The answer for `memory_gb` of memory.
pub fn row(memory_gb: f64, unified: bool) -> Row {
    let usable = usable_bytes(memory_gb, unified);
    Row {
        memory_gb,
        usable,
        infer: Precision::ALL.map(|p| max_params_inference(usable, p)),
        train: max_params_training(usable),
    }
}
