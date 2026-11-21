// Provenance: the "one GGUF file holding p{i}.data / p{i}.m / p{i}.v plus step and run keys"
// idea follows ferrum-cpu's training checkpoints (`train_transformer/checkpoint.rs`, commit
// 56a9d3a). The kind below is our own and is NOT interchangeable with ferrum-cpu's checkpoints.
//! Training checkpoints as GGUF files: weights, Adam moments, step and loop position.

use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value};
use ferrum_ref::gpt2::{param_lens, Config, Gpt2};
use ferrum_ref::trainer::{Checkpoint, RunShape, Snapshot};
use std::io::Write;
use std::path::Path;

/// `general.architecture` of a checkpoint file.
pub const KIND: &str = "ferrum-gpu-checkpoint";

fn err(m: impl Into<String>) -> GgufError {
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

fn get(g: &Gguf, k: &str) -> Result<u64, GgufError> {
    g.uint(&format!("ferrum.ckpt.{k}")).map(|v| v as u64)
}

/// Parse a checkpoint. Every size is derived from the config with checked arithmetic and
/// compared with the file's own tensor directory before any model-sized allocation.
pub fn from_bytes(bytes: Vec<u8>) -> Result<Checkpoint, GgufError> {
    let g = Gguf::parse(bytes)?;
    let kind = g.string("general.architecture")?;
    if kind != KIND {
        return Err(err(format!("not a ferrum-gpu checkpoint (kind '{kind}')")));
    }
    let dim = |k: &str| {
        get(&g, k).and_then(|v| usize::try_from(v).map_err(|_| err(format!("'{k}' is too large"))))
    };
    let cfg = Config {
        vocab: dim("vocab")?,
        ctx: dim("ctx")?,
        dim: dim("dim")?,
        heads: dim("heads")?,
        layers: dim("layers")?,
        hidden: dim("hidden")?,
    };
    let lens = param_lens(&cfg).ok_or_else(|| err("the checkpoint's model config is invalid"))?;
    if g.tensors.len()
        != lens
            .len()
            .checked_mul(3)
            .ok_or_else(|| err("tensor count overflows"))?
    {
        return Err(err("the tensor count does not match the model config"));
    }
    for (i, &n) in lens.iter().enumerate() {
        for kind in ["data", "m", "v"] {
            let name = format!("p{i}.{kind}");
            let t = g
                .tensor(&name)
                .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
            if t.ggml_type != ferrum_gguf::GGML_F32 || t.dims != [n as u64] {
                return Err(err(format!(
                    "tensor '{name}' must be F32 with {n} elements"
                )));
            }
        }
    }
    let load = |kind: &str| -> Result<Gpt2, GgufError> {
        let params = (0..lens.len())
            .map(|i| g.tensor_f32(&format!("p{i}.{kind}")))
            .collect::<Result<Vec<_>, _>>()?;
        Gpt2::from_params(cfg, params).map_err(err)
    };
    Ok(Checkpoint {
        snapshot: Snapshot {
            weights: load("data")?,
            m: load("m")?,
            v: load("v")?,
            steps: get(&g, "step")?,
        },
        epoch: get(&g, "epoch")?,
        step_in_epoch: get(&g, "step_in_epoch")?,
        shape: RunShape {
            context: get(&g, "shape.context")?,
            batch: get(&g, "shape.batch")?,
            stride: get(&g, "shape.stride")?,
            windows: get(&g, "shape.windows")?,
            seed: get(&g, "shape.seed")?,
        },
        corpus_id: g.string("ferrum.ckpt.corpus_id").ok().map(String::from),
    })
}

/// Write `bytes` to `path` atomically: a temp file in the same directory, flushed, then renamed
/// over the target, so a crash leaves either the old file or the new one, never half of one.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
    })?;
    let tmp = path.with_file_name(format!(
        ".{}.tmp{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    let result = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}
#[cfg(test)]
mod tests;
