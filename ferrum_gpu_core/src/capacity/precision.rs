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
