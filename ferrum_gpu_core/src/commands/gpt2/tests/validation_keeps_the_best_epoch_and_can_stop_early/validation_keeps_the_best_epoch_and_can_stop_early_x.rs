use super::*;

#[test]
fn validation_keeps_the_best_epoch_and_can_stop_early() {
    let d = dir("val");
    let (c, m) = (data(&d), d.join("m.gguf"));
    let t = ok(&format!(
        "train {FAST} --epochs 12 --data {c} --out {} --val 0.2 --patience 2 --lr 0.5 --quant f32",
        m.display()
    ));
    assert!(t.contains("val loss") && t.contains("(best)"), "{t}");
    assert!(
        t.contains("stopped early: no validation improvement") || t.contains("steps: 12"),
        "{t}"
    );
    assert!(err(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --val 0",
        m.display()
    ))
    .contains("--val must be"));
    assert!(err(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --val x",
        m.display()
    ))
    .contains("not a number"));
}

#[test]
fn checkpoint_and_resume_continue_the_same_run() {
    let d = dir("resume");
    let (c, ck, a, b) = (
        data(&d),
        d.join("ck.gguf"),
        d.join("a.gguf"),
        d.join("b.gguf"),
    );
    let tok = d.join("tok.gguf");
    let base = format!(
        "train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --data {c} --tokenizer {} --quant f32",
        tok.display()
    );
    ok(&format!(
        "{base} --epochs 6 --out {} --checkpoint {}",
        a.display(),
        ck.display()
    ));
    let r = ok(&format!(
        "{base} --epochs 10 --out {} --resume {}",
        b.display(),
        ck.display()
    ));
    assert!(r.contains("resumed at step 42"), "{r}");
    // an uninterrupted run of 10 epochs ends at the same weights
    let full = d.join("full.gguf");
    ok(&format!("{base} --epochs 10 --out {}", full.display()));
    assert_eq!(std::fs::read(&b).unwrap(), std::fs::read(&full).unwrap());
    // a different tokenizer is refused
    let other = d.join("other.gguf");
    let c2 = d.join("c2.txt");
    std::fs::write(&c2, "dog dig dug dig dog bog bug. ".repeat(60)).unwrap();
    let c2 = c2.display().to_string();
    ok(&format!("train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --data {c2} --tokenizer {} --vocab 280 --epochs 1 --out {}", other.display(), d.join("o.gguf").display()));
    let bad = err(&format!("train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --data {c} --tokenizer {} --epochs 10 --out {} --resume {}", other.display(), b.display(), ck.display()));
    assert!(
        bad.contains("different tokenizer") || bad.contains("tokenizer has"),
        "{bad}"
    );
    assert!(err(&format!(
        "{base} --epochs 3 --out {} --resume /nope",
        b.display()
    ))
    .contains("/nope"));
}
