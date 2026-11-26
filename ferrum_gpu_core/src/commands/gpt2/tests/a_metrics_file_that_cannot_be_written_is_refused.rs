use super::*;

#[test]
#[cfg(target_os = "linux")]
fn a_metrics_file_that_cannot_be_written_is_refused() {
    let d = dir("metrics_full");
    let c = data(&d);
    // /dev/full opens but refuses every write, so the header row fails like a full disk
    let e = err(&format!(
        "train {FAST} --epochs 1 --data {c} --out {} --metrics-csv /dev/full",
        d.join("m.gguf").display()
    ));
    assert!(e.starts_with("/dev/full: "), "{e}");
}
