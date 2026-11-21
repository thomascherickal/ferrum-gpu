use super::*;

/// `general.architecture` of a checkpoint file.
pub const KIND: &str = "ferrum-gpu-checkpoint";

pub(super) fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// Serialise a checkpoint.
pub fn to_bytes(cp: &Checkpoint) -> Vec<u8> {
    let c = cp.snapshot.weights.cfg;
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(KIND.into()));
    for (k, v) in [
        ("vocab", c.vocab),
        ("ctx", c.ctx),
        ("dim", c.dim),
        ("heads", c.heads),
        ("layers", c.layers),
        ("hidden", c.hidden),
    ] {
        b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v as u64));
    }
    for (k, v) in [
        ("step", cp.snapshot.steps),
        ("epoch", cp.epoch),
        ("step_in_epoch", cp.step_in_epoch),
        ("shape.context", cp.shape.context),
        ("shape.batch", cp.shape.batch),
        ("shape.stride", cp.shape.stride),
        ("shape.windows", cp.shape.windows),
        ("shape.seed", cp.shape.seed),
    ] {
        b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
    }
    if let Some(id) = &cp.corpus_id {
        b.meta("ferrum.ckpt.corpus_id", Value::Str(id.clone()));
    }
    let (w, m, v) = (
        cp.snapshot.weights.params(),
        cp.snapshot.m.params(),
        cp.snapshot.v.params(),
    );
    for i in 0..w.len() {
        for (kind, set) in [("data", &w), ("m", &m), ("v", &v)] {
            b.tensor_f32(&format!("p{i}.{kind}"), &[set[i].len() as u64], set[i]);
        }
    }
    b.into_bytes()
}

pub(super) fn get(g: &Gguf, k: &str) -> Result<u64, GgufError> {
    g.uint(&format!("ferrum.ckpt.{k}")).map(|v| v as u64)
}
