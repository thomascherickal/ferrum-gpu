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

mod a_checkpoint_file_on_disk_resumes_a_run;
mod a_config_that_lies_about_sizes_is_refused_before_allocating;
mod atomic_write_replaces_the_target_and_leaves_no_temp_file;
mod round_trips_exactly;
