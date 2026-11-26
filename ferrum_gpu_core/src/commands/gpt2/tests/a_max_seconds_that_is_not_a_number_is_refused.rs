use super::*;

#[test]
fn a_max_seconds_that_is_not_a_number_is_refused() {
    let d = dir("max_seconds_nan");
    let c = data(&d);
    let e = err(&format!(
        "train {FAST} --epochs 1 --data {c} --out {} --max-seconds soon",
        d.join("m.gguf").display()
    ));
    assert!(e.contains("--max-seconds"), "{e}");
}
