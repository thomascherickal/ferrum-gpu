use super::*;

#[test]
fn a_checkpoint_file_on_disk_resumes_a_run() {
    use ferrum_ref::trainer::{train, CpuTrainer, Engine, Hooks, NoHooks, StepStats, TrainConfig};
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
