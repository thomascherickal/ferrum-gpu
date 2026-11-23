//! Outputs that cannot be written: the parent is missing or the target is a directory. Each
//! failure names the path.

use super::{err, scratch, FAST};

fn train(d: &std::path::Path, c: &std::path::Path, extra: &str) -> String {
    err(&format!(
        "train {FAST} --epochs 1 --vocab 300 --data {} --out {} {extra}",
        c.display(),
        d.join("o.gguf").display()
    ))
}

#[test]
fn train_names_each_output_it_cannot_write() {
    let (d, c) = scratch("unwritable");
    let missing = "/nonexistent-ferrum-dir/x";
    let e = train(&d, &c, &format!("--metrics-csv {}", d.display()));
    assert!(e.starts_with(&format!("{}: ", d.display())), "{e}");
    let e = train(
        &d,
        &c,
        &format!("--checkpoint {missing} --checkpoint-every 1"),
    );
    assert!(e.starts_with(&format!("{missing}: ")), "{e}");
    let e = train(&d, &c, &format!("--tokenizer {missing}"));
    assert!(e.starts_with(&format!("{missing}: ")), "{e}");
    let e = err(&format!(
        "train {FAST} --epochs 1 --vocab 300 --data {} --out {missing}",
        c.display()
    ));
    assert!(e.starts_with(&format!("{missing}: ")), "{e}");
}
