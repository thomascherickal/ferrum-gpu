use super::*;

#[test]
pub(super) fn a_diverging_run_stops_with_advice_not_nan_weights() {
    let (d, corpus) = scratch("diverge");
    let out = d.join("o.gguf");
    let e = err(&format!(
        "train {FAST} --epochs 6 --vocab 262 --lr 1e30 --clip 0 --no-qat --data {} --out {}",
        corpus.display(),
        out.display()
    ));
    assert!(
        e.contains("the loss became") && e.contains("lower --lr"),
        "{e}"
    );
    assert!(!out.exists(), "no model is written from a diverged run");
}

#[cfg(target_os = "linux")]
#[test]
pub(super) fn a_metrics_file_that_cannot_be_written_stops_the_run_by_name() {
    // /dev/full accepts the open and fails every write, starting with the header
    let (d, corpus) = scratch("csvfull");
    let e = err(&format!(
        "train {FAST} --epochs 2 --vocab 262 --data {} --out {} --metrics-csv /dev/full",
        corpus.display(),
        d.join("o.gguf").display()
    ));
    assert!(
        e.contains("/dev/full") && e.contains("No space left"),
        "{e}"
    );
}

#[test]
pub(super) fn finetune_refuses_a_gpt2_only_flag_for_a_llama_file() {
    let (d, corpus) = scratch("ftllama");
    let llama = d.join("l.gguf");
    ok(&format!(
        "train --device cpu --arch llama --preset tiny --steps 3 --batch 2 --context 16 --data {} --out {}",
        corpus.display(),
        llama.display()
    ));
    let e = err(&format!(
        "finetune-gguf --device cpu --model {} --data {} --out {} --warmup 2",
        llama.display(),
        corpus.display(),
        d.join("o.gguf").display()
    ));
    assert!(e.contains("--warmup needs a gpt2 model"), "{e}");
}

#[test]
pub(super) fn resuming_with_another_run_shape_is_refused_and_resuming_a_finished_run_trains_nothing(
) {
    let (d, corpus) = scratch("resume_shape");
    let (ck, tok) = (d.join("ck.ckpt"), d.join("t.gguf"));
    let first = d.join("a.gguf");
    ok(&format!(
        "train {FAST} --epochs 2 --vocab 262 --data {} --out {} --tokenizer {} --checkpoint {}",
        corpus.display(),
        first.display(),
        tok.display(),
        ck.display()
    ));
    // the same run with another batch is a different run: the checkpoint is not for it
    let e = err(&format!(
        "train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 8 --epochs 2 --vocab 262 --data {} --out {} --tokenizer {} --resume {}",
        corpus.display(),
        d.join("b.gguf").display(),
        tok.display(),
        ck.display()
    ));
    assert!(e.contains("resume"), "{e}");
    // the same run again: the checkpoint is already at the end, so there is nothing left to train
    let o = ok(&format!(
        "train {FAST} --epochs 2 --vocab 262 --data {} --out {} --tokenizer {} --resume {}",
        corpus.display(),
        d.join("c.gguf").display(),
        tok.display(),
        ck.display()
    ));
    assert!(o.contains("resumed at step"), "{o}");
    assert!(!o.contains("steps:"), "{o}");
}
