use super::*;

#[test]
fn train_writes_a_gpt2_file_with_its_tokenizer_and_generate_uses_it() {
    let d = dir("roundtrip");
    let (c, m, tok) = (data(&d), d.join("m.gguf"), d.join("tok.gguf"));
    let t = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 300 --tokenizer {}",
        m.display(),
        tok.display()
    ));
    assert!(
        t.contains("tokenizer: trained") && t.contains("steps: 21") && t.contains("wrote "),
        "{t}"
    );
    assert!(is_gpt2(m.to_str().unwrap()) && !is_gpt2("/nope"));
    // a second run reuses the saved tokenizer file
    let t2 = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --tokenizer {}",
        m.display(),
        tok.display()
    ));
    assert!(
        t2.contains("tokenizer: ") && !t2.contains("trained"),
        "{t2}"
    );
    let g = ok(&format!(
        "generate --device cpu --model {} --prompt the --max-new 8",
        m.display()
    ));
    assert!(g.starts_with("the"), "{g}");
    assert!(ok(&format!("info --device cpu --model {}", m.display())).contains("model: gpt2 vocab"));
    let r = ok(&format!(
        "run-gguf --device cpu --model {} --prompt the --max-new 2",
        m.display()
    ));
    assert!(
        r.contains("device: CPU") && r.contains("model: gpt2"),
        "{r}"
    );
}

#[test]
fn generate_options_stop_stream_ids_and_aliases() {
    let d = dir("gen");
    let (c, m) = (data(&d), d.join("m.gguf"));
    ok(&format!(
        "train {FAST} --epochs 40 --data {c} --out {} ",
        m.display()
    ));
    let g = |extra: &str| {
        ok(&format!(
            "generate --device cpu --model {} {extra}",
            m.display()
        ))
    };
    let base = g("--prompt the --max-new 30");
    // the same run with cpu's spellings gives the same text
    assert_eq!(base, g("--prompt the --max-tokens 30"));
    assert_eq!(base, g("--prompt the --chars 30"));
    assert_eq!(g("--prompt the --max-new 30 --temp 0 --gen-seed 5"), base);
    // a stop string ends the text before it
    let stopped = g("--prompt the --max-new 30 --stop he");
    assert!(
        stopped.len() < base.len() && !stopped["the".len()..].contains("he"),
        "{stopped}"
    );
    // streaming prints the same text, then a newline
    assert_eq!(g("--prompt the --max-new 30 --stream"), base);
    // sampling knobs run and are seeded
    let s1 = g("--prompt the --max-new 20 --temp 1 --top-p 0.9 --top-k 5 --repetition-penalty 1.2 --seed 3");
    assert_eq!(s1, g("--prompt the --max-new 20 --temp 1 --top-p 0.9 --top-k 5 --repetition-penalty 1.2 --seed 3"));
    // raw ids instead of a prompt
    let mut out = Vec::new();
    let by_ids = words(&format!(
        "generate --device cpu --model {} --max-new 3",
        m.display()
    ));
    let by_ids: Vec<String> = by_ids
        .into_iter()
        .chain(["--ids".into(), "100 101".into()])
        .collect();
    run(&by_ids, &mut out).unwrap();
    assert!(!out.is_empty());
    assert!(run(
        &words(&format!(
            "generate --device cpu --model {} --ids 1x",
            m.display()
        )),
        &mut Vec::new()
    )
    .unwrap_err()
    .contains("not an id"));
    assert!(err(&format!(
        "generate --device cpu --model {} --prompt {}",
        m.display(),
        "a".repeat(400)
    ))
    .contains("context"));
    let empty: Vec<String> = [
        "generate",
        "--device",
        "cpu",
        "--model",
        m.to_str().unwrap(),
        "--prompt",
        "",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    assert!(run(&empty, &mut Vec::new())
        .unwrap_err()
        .contains("prompt is empty"));
}

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
