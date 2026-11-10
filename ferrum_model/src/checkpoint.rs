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
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            vocab: 9,
            ctx: 6,
            dim: 8,
            heads: 2,
            layers: 2,
            hidden: 12,
        }
    }

    fn sample() -> Checkpoint {
        let w = Gpt2::random(cfg(), 1);
        Checkpoint {
            snapshot: Snapshot {
                weights: w.clone(),
                m: Gpt2::random(cfg(), 2),
                v: Gpt2::random(cfg(), 3),
                steps: 42,
            },
            epoch: 3,
            step_in_epoch: 5,
            shape: RunShape {
                context: 6,
                batch: 4,
                stride: 6,
                windows: 100,
                seed: 1337,
            },
            corpus_id: Some("sha256:abc".into()),
        }
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_gpu_ckpt_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trips_exactly() {
        let cp = sample();
        assert_eq!(from_bytes(to_bytes(&cp)).unwrap(), cp);
        let mut no_id = cp.clone();
        no_id.corpus_id = None;
        assert_eq!(from_bytes(to_bytes(&no_id)).unwrap(), no_id);
    }

    #[test]
    fn names_follow_the_p_i_data_m_v_scheme() {
        let g = Gguf::parse(to_bytes(&sample())).unwrap();
        assert_eq!(g.string("general.architecture").unwrap(), KIND);
        let n = param_lens(&cfg()).unwrap().len();
        assert_eq!(g.tensors.len(), 3 * n);
        for name in ["p0.data", "p0.m", "p0.v", &format!("p{}.v", n - 1)] {
            assert!(g.tensor(name).is_some(), "{name}");
        }
        assert_eq!(g.uint("ferrum.ckpt.step").unwrap(), 42);
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let good = to_bytes(&sample());
        // Cutting only the final alignment padding leaves a valid file; stop before real data ends.
        let cuts = [0, 1, 10, 100, 1000, good.len() / 2, good.len() - 200];
        for cut in cuts {
            assert!(from_bytes(good[..cut].to_vec()).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn wrong_kind_and_missing_keys_are_named() {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("gpt2".into()));
        assert!(from_bytes(b.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("not a ferrum-gpu checkpoint"));
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str(KIND.into()));
        assert!(from_bytes(b.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("missing key"));
    }

    #[test]
    fn a_config_that_lies_about_sizes_is_refused_before_allocating() {
        // Same tensors, but the metadata claims a vastly larger model.
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str(KIND.into()));
        for (k, v) in [
            ("vocab", u64::MAX / 4),
            ("ctx", 6),
            ("dim", 8),
            ("heads", 2),
            ("layers", 2),
            ("hidden", 12),
        ] {
            b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
        }
        assert!(from_bytes(b.into_bytes()).is_err());
        // Plausible config, but the file carries no tensors.
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str(KIND.into()));
        for (k, v) in [
            ("vocab", 9u64),
            ("ctx", 6),
            ("dim", 8),
            ("heads", 2),
            ("layers", 2),
            ("hidden", 12),
        ] {
            b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
        }
        assert!(from_bytes(b.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("tensor count"));
        // Invalid config (heads do not divide dim).
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str(KIND.into()));
        for (k, v) in [
            ("vocab", 9u64),
            ("ctx", 6),
            ("dim", 8),
            ("heads", 3),
            ("layers", 2),
            ("hidden", 12),
        ] {
            b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
        }
        assert!(from_bytes(b.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("invalid"));
    }

    #[test]
    fn a_tensor_of_the_wrong_length_is_named() {
        let cp = sample();
        let good = Gguf::parse(to_bytes(&cp)).unwrap();
        // Rebuild the file but shorten one moment tensor.
        let mut b = GgufBuilder::new();
        for (k, v) in &good.kv {
            b.meta(k, v.clone());
        }
        for t in &good.tensors {
            let mut data = good.tensor_f32(&t.name).unwrap();
            if t.name == "p4.m" {
                data.pop();
            }
            b.tensor_f32(&t.name, &[data.len() as u64], &data);
        }
        let e = from_bytes(b.into_bytes()).unwrap_err().to_string();
        assert!(e.contains("p4.m") && e.contains("elements"), "{e}");
    }

    #[test]
    fn atomic_write_replaces_the_target_and_leaves_no_temp_file() {
        let d = tmpdir("atomic");
        let p = d.join("run.ckpt");
        atomic_write(&p, b"first").unwrap();
        atomic_write(&p, b"second, longer").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"second, longer");
        let names: Vec<_> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "no stray temp file: {names:?}");
        assert!(atomic_write(&d.join("no/such/dir/x"), b"x").is_err());
        assert!(atomic_write(Path::new("/"), b"x").is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_checkpoint_file_on_disk_resumes_a_run() {
        use ferrum_ref::trainer::{
            train, CpuTrainer, Engine, Hooks, NoHooks, StepStats, TrainConfig,
        };
        let c = cfg();
        let toks: Vec<usize> = (0..300)
            .map(|i| [3usize, 7, 1, 8, 4, 2, 5, 6][i % 8])
            .collect();
        let tc = TrainConfig {
            context: 6,
            batch: 5,
            max_steps: 12,
            lr: 0.01,
            clip: 0.5,
            seed: 4,
            ..TrainConfig::default()
        };
        let mut straight = CpuTrainer::new(Gpt2::init(c, 8), 1);
        train(&mut straight, &toks, &tc, None, None, &mut NoHooks).unwrap();

        let d = tmpdir("resume");
        let path = d.join("run.ckpt");
        struct ToDisk(std::path::PathBuf, bool);
        impl Hooks for ToDisk {
            fn on_step(&mut self, s: u64, _: &StepStats) -> bool {
                !(self.1 && s == 5)
            }
            fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
                atomic_write(&self.0, &to_bytes(cp)).map_err(|e| e.to_string())
            }
        }
        let mut a = CpuTrainer::new(Gpt2::init(c, 8), 1);
        let r = train(
            &mut a,
            &toks,
            &tc,
            None,
            Some("corpus-1"),
            &mut ToDisk(path.clone(), true),
        )
        .unwrap();
        assert!(r.stopped);
        // A different "process": read the file, restore, continue.
        let cp = from_bytes(std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            (cp.snapshot.steps, cp.corpus_id.as_deref()),
            (5, Some("corpus-1"))
        );
        let mut b = CpuTrainer::new(Gpt2::init(c, 12345), 1);
        b.restore(&cp.snapshot).unwrap();
        train(&mut b, &toks, &tc, Some(&cp.shape), None, &mut NoHooks).unwrap();
        assert_eq!(b.snapshot().unwrap(), straight.snapshot().unwrap());
        let _ = std::fs::remove_dir_all(&d);
    }
}
