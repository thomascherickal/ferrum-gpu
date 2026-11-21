use super::*;

#[test]
fn metrics_csv_verbose_stop_file_and_max_seconds() {
    let d = dir("hooks");
    let (c, m, csv) = (data(&d), d.join("m.gguf"), d.join("m.csv"));
    let t = ok(&format!("train {FAST} --epochs 4 --data {c} --out {} --metrics-csv {} --verbose --cosine --warmup 2 --clip 1 --weight_decay 0.01", m.display(), csv.display()));
    assert!(t.contains("step 10/28") && t.contains("tokens/s"), "{t}");
    let rows: Vec<String> = std::fs::read_to_string(&csv)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    assert_eq!(
        rows[0],
        "step,epoch,loss,lr,grad_norm,tokens_per_sec,elapsed"
    );
    assert_eq!(rows.len(), 29);
    assert!(rows[1].starts_with("1,0,"));
    // cosine: the learning rate changes over the run
    let lr = |r: &str| r.split(',').nth(3).unwrap().to_string();
    assert_ne!(lr(&rows[1]), lr(&rows[28]));
    // a stop file ends the run after one step
    std::fs::write(d.join("s.stop"), "").unwrap();
    let s = ok(&format!(
        "train {FAST} --epochs 100 --data {c} --out {}",
        d.join("s.gguf").display()
    ));
    assert!(
        s.contains("stopped early") && s.contains("steps: 1 "),
        "{s}"
    );
    // --max-seconds 0 stops after the first step
    let z = ok(&format!(
        "train {FAST} --epochs 100 --data {c} --out {} --max-seconds 0",
        d.join("z.gguf").display()
    ));
    assert!(z.contains("--max-seconds reached"), "{z}");
}
