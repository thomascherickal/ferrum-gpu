//! Every place a command writes to its output can fail (a closed pipe, a full disk). For each
//! command, run it with the output failing at the first write, then at the second, and so on until
//! the command finishes: each run must report that failure and nothing else, so every `say(...)?`
//! of the command has been the one that failed.

use super::{scratch, trained, words, FailNth, FAST};
use crate::commands::run;
use std::io::Write;

/// Run `cmd` with the output failing from write `n` for every `n` until it succeeds; returns how
/// many writes the successful run made (so a command that never writes is noticed).
fn sweep(cmd: &str) -> usize {
    for n in 0..400 {
        match run(&words(cmd), &mut FailNth::new(n)) {
            Ok(()) => return n,
            Err(e) => assert!(
                e.contains("output closed"),
                "write {n} of `{cmd}` failed for another reason: {e}"
            ),
        }
    }
    panic!("`{cmd}` still writes after 400 writes");
}

#[test]
fn the_failing_writer_flushes_and_counts_its_writes() {
    let mut w = FailNth::new(1);
    assert!(w.write(b"a").is_ok() && w.write(b"b").is_err());
    assert!(w.flush().is_ok());
    let mut f = super::FailAfter { budget: 1 };
    assert!(f.write(b"xy").is_ok() && f.write(b"z").is_err());
    assert!(f.flush().is_ok());
}

#[test]
fn info_generate_eval_and_export_report_a_closed_output_at_every_write() {
    let (d, c) = scratch("sweep_read");
    let (m, t) = trained(&d, &c, 300, "m");
    let (m, c, t, d) = (m.display(), c.display(), t.display(), d.display());
    assert!(sweep("info --device cpu") > 0);
    assert!(sweep(&format!("info --device cpu --model {m}")) > 0);
    assert!(sweep(&format!("generate --device cpu --model {m} --prompt the --max-new 3")) > 0);
    assert!(sweep(&format!("run-gguf --device cpu --model {m} --prompt the --max-new 3 --stream")) > 0);
    assert!(sweep(&format!("eval --model {m} --data {c} --json {d}/e.json")) > 0);
    assert!(sweep(&format!("export-gguf --model {m} --out {d}/q.gguf --quant int8")) > 0);
    assert!(sweep(&format!("tokenize --data {c} --tokenizer {t} --out {d}/c.tok.gguf")) > 0);
    assert!(sweep(&format!("prepare --input {c} --out {d}/prep --val-frac 0.2 --test-frac 0.2 --dedup")) > 0);
}

#[test]
fn training_reports_a_closed_output_at_every_write() {
    let (d, c) = scratch("sweep_train");
    let (c, d) = (c.display(), d.display());
    let gpt2 = format!(
        "train {FAST} --epochs 2 --vocab 300 --val 0.2 --patience 1 --sample --verbose --data {c} --out {d}/g.gguf --tokenizer {d}/g.tok.gguf"
    );
    assert!(sweep(&gpt2) > 0);
    let llama = format!("train --arch llama --device cpu --preset tiny --steps 3 --batch 2 --context 16 --verbose --data {c} --out {d}/l.gguf");
    assert!(sweep(&llama) > 0);
    let tune = format!("finetune-gguf --model {d}/g.gguf --device cpu --epochs 1 --batch 4 --data {c} --out {d}/f.gguf");
    assert!(sweep(&tune) > 0);
}
