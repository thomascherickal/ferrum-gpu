//! What fits: the largest model a given amount of memory can hold, and how fast it could decode.
//! These are the back-of-the-envelope formulas of `hardware-caps.md`, in Rust, so the Capable tab and the
//! document agree: `scripts/capability-estimates.py` prints the same table and a test compares them.
//! Everything here is an **estimate**: the memory formula is within -24% to +37% of the measured plans
//! (capability.md), and the speed ceiling is physics (bandwidth over weight bytes), not what ferrum-gpu reaches.

use crate::json::Json;

/// Bytes per weight of each precision (a 32-weight block carries its scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// 4 bytes.
    F32,
    /// 34/32 bytes.
    Q8_0,
    /// 18/32 bytes.
    Q4_0,
}

impl Precision {
    /// All three, in the order the tables list them.
    pub const ALL: [Precision; 3] = [Precision::Q4_0, Precision::Q8_0, Precision::F32];

    /// Bytes per weight.
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            Precision::F32 => 4.0,
            Precision::Q8_0 => 34.0 / 32.0,
            Precision::Q4_0 => 18.0 / 32.0,
        }
    }

    /// The name shown to the user.
    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::Q8_0 => "Q8_0",
            Precision::Q4_0 => "Q4_0",
        }
    }
}

/// Share of a discrete card's memory a model may use (display and driver take the rest).
pub const USABLE_DISCRETE: f64 = 0.90;
/// Share of unified memory a model may use (an assumption: the system and the application take the rest).
pub const USABLE_UNIFIED: f64 = 0.70;
/// Fixed overhead of inference (a 2048-token KV cache in f32, the f32 embedding and scratch of a small model).
pub const OVERHEAD_BYTES: f64 = 300e6;
/// Extra weight memory of inference over the packed weights (the f32 parts that scale with the model).
pub const INFER_FACTOR: f64 = 1.10;
/// Bytes per parameter of full fine-tuning in f32 with AdamW: weights, gradients and two moments.
pub const TRAIN_BYTES_PER_PARAM: f64 = 16.0;
/// Extra memory of training over those 16 bytes (activations at a small batch; measured 2.6% to 8%).
pub const TRAIN_FACTOR: f64 = 1.08;

/// The memory an inference of `params` weights at `p` needs, in bytes.
pub fn inference_bytes(params: f64, p: Precision) -> f64 {
    params * p.bytes_per_weight() * INFER_FACTOR + OVERHEAD_BYTES
}

/// The memory a full f32 fine-tune of `params` weights needs, in bytes.
pub fn training_bytes(params: f64) -> f64 {
    params * TRAIN_BYTES_PER_PARAM * TRAIN_FACTOR + OVERHEAD_BYTES
}

/// The most parameters inference at `p` can hold in `usable` bytes (0 if even the overhead does not fit).
pub fn max_params_inference(usable: f64, p: Precision) -> f64 {
    ((usable - OVERHEAD_BYTES) / (p.bytes_per_weight() * INFER_FACTOR)).max(0.0)
}

/// The most parameters full f32 fine-tuning can hold in `usable` bytes.
pub fn max_params_training(usable: f64) -> f64 {
    ((usable - OVERHEAD_BYTES) / (TRAIN_BYTES_PER_PARAM * TRAIN_FACTOR)).max(0.0)
}

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

/// The JSON the page reads for a row.
pub fn row_json(r: &Row) -> Json {
    let infer = Precision::ALL
        .iter()
        .zip(r.infer)
        .map(|(p, v)| (p.name(), v))
        .collect::<Vec<_>>();
    Json::obj([
        ("memory_gb", Json::Num(r.memory_gb)),
        ("usable_gb", Json::Num(r.usable / 1e9)),
        (
            "infer",
            Json::Arr(
                infer
                    .iter()
                    .map(|(n, v)| {
                        Json::obj([
                            ("precision", Json::Str((*n).into())),
                            ("params", Json::Num(*v)),
                            ("text", Json::Str(format_params(*v))),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "train",
            Json::obj([
                ("params", Json::Num(r.train)),
                ("text", Json::Str(format_params(r.train))),
            ]),
        ),
    ])
}

#[cfg(test)]
mod tests;
