use super::*;
use crate::cmds::run;
use std::path::PathBuf;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn ok(s: &str) -> String {
    let mut out = Vec::new();
    run(&words(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

fn err(s: &str) -> String {
    run(&words(s), &mut Vec::new()).unwrap_err()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gpt2_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn data(d: &std::path::Path) -> String {
    let p = d.join("c.txt");
    std::fs::write(&p, "the cat sat on the mat. ".repeat(60)).unwrap();
    p.display().to_string()
}

const FAST: &str =
    "--device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4";

#[test]
fn sizes_follow_ferrum_cpu_defaults_presets_and_explicit_flags() {
    let table = [&SIZE_FLAGS[..]].concat();
    let parse = |s: &str| Args::parse(&table, &words(s)).unwrap();
    let d = Sizes::from_flags(&parse("")).unwrap();
    assert_eq!(d, Sizes::DEFAULT);
    let small = Sizes::from_flags(&parse("--preset small")).unwrap();
    assert_eq!(
        (
            small.ctx,
            small.dim,
            small.layers,
            small.hidden,
            small.vocab
        ),
        (128, 128, 4, 512, 2048)
    );
    assert_eq!(small.lr, 3e-3);
    for order in [
        "--preset tiny --embed 32 --blocks 3",
        "--embed 32 --blocks 3 --preset tiny",
    ] {
        let s = Sizes::from_flags(&parse(order)).unwrap();
        assert_eq!(
            (s.dim, s.layers, s.ctx, s.hidden, s.vocab),
            (32, 3, 64, 256, 512),
            "{order}"
        );
    }
    assert!(Sizes::from_flags(&parse("--preset huge"))
        .unwrap_err()
        .contains("valid: tiny, small, medium"));
    assert!(Sizes::from_flags(&parse("--embed x"))
        .unwrap_err()
        .contains("--embed"));
}

#[test]
fn quant_accepts_both_spellings() {
    let t = [crate::flags::val("--quant")];
    let q = |s: &str| quant(&Args::parse(&t, &words(s)).unwrap(), "int8");
    assert_eq!(q("--quant int4").unwrap(), Quant::Q4_0);
    assert_eq!(q("--quant q4_0").unwrap(), Quant::Q4_0);
    assert_eq!(q("--quant int8").unwrap(), Quant::Q8_0);
    assert_eq!(q("--quant f32").unwrap(), Quant::F32);
    assert_eq!(q("").unwrap(), Quant::Q8_0);
    assert!(q("--quant q2").unwrap_err().contains("unknown"));
}

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

#[test]
fn finetune_continues_from_a_gpt2_file_with_its_tokenizer() {
    let d = dir("ft");
    let (c, m, g) = (data(&d), d.join("m.gguf"), d.join("g.gguf"));
    ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {}",
        m.display()
    ));
    let t = ok(&format!(
        "finetune-gguf --device cpu --model {} --data {c} --out {} --epochs 2 --batch 4",
        m.display(),
        g.display()
    ));
    assert!(t.contains("steps: 14") && !t.contains("tokenizer:"), "{t}");
    assert!(is_gpt2(g.to_str().unwrap()));
    assert!(err(&format!(
        "finetune-gguf --device cpu --model {} --data {c} --out x",
        "/nope"
    ))
    .contains("/nope"));
}

#[test]
fn eval_reports_loss_and_writes_json() {
    let d = dir("eval");
    let (c, m, j) = (data(&d), d.join("m.gguf"), d.join("e.json"));
    ok(&format!(
        "train {FAST} --epochs 20 --data {c} --out {}",
        m.display()
    ));
    let e = ok(&format!(
        "eval --model {} --data {c} --ctx 16 --json {}",
        m.display(),
        j.display()
    ));
    assert!(e.contains("tokens: ") && e.contains("perplexity: "), "{e}");
    let json = std::fs::read_to_string(&j).unwrap();
    assert!(
        json.starts_with("{\"tokens\":") && json.contains("\"perplexity\":"),
        "{json}"
    );
    assert!(err(&format!(
        "eval --model {} --data {}",
        m.display(),
        d.join("tiny.txt").display()
    ))
    .contains("--data"));
    std::fs::write(d.join("one.txt"), "a").unwrap();
    assert!(err(&format!(
        "eval --model {} --data {}",
        m.display(),
        d.join("one.txt").display()
    ))
    .contains("at least 2"));
}

#[test]
fn unsupported_and_misplaced_options_are_refused_by_name() {
    let d = dir("refuse");
    let (c, o) = (data(&d), d.join("o.gguf").display().to_string());
    let t = |extra: &str| err(&format!("train --device cpu --data {c} --out {o} {extra}"));
    assert!(t("--preset tiny --tie").contains("--tie is not supported"));
    assert!(t("--preset tiny --dropout 0.1").contains("--dropout is not supported"));
    assert!(t("--preset tiny --vocab 0").contains("character-level"));
    assert!(t("--preset tiny --arch bert").contains("unknown '"));
    assert!(t("--preset tiny --arch llama --epochs 3").contains("--epochs needs --arch gpt2"));
    assert!(t("--preset tiny --arch llama --embed 8").contains("--embed needs --arch gpt2"));
    assert!(t("--preset huge").contains("unknown --preset"));
    assert!(t("--preset tiny --vocab 100").contains("tokenizer"));
    assert!(err(&format!("train --device cpu --data /nope --out {o}")).contains("--data"));
    let bin = d.join("bin.txt");
    std::fs::write(&bin, [0xffu8, 0xfe, 0x00]).unwrap();
    assert!(err(&format!(
        "train --device cpu --data {} --out {o}",
        bin.display()
    ))
    .contains("not valid UTF-8"));
}

#[test]
fn a_file_without_a_tokenizer_but_with_256_tokens_uses_bytes() {
    use ferrum_model::gpt2_gguf::to_gguf;
    use ferrum_ref::gpt2::{Config, Gpt2};
    let d = dir("bytes");
    let cfg = Config {
        vocab: 256,
        ctx: 16,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    };
    let p = d.join("b.gguf");
    std::fs::write(&p, to_gguf(&Gpt2::init(cfg, 1), Quant::F32)).unwrap();
    let l = load(p.to_str().unwrap()).unwrap();
    assert_eq!(l.tok.vocab_size(), 256);
    assert!(ok(&format!(
        "generate --device cpu --model {} --prompt ab --max-new 3",
        p.display()
    ))
    .starts_with("ab"));
    // 100 tokens with no stored tokenizer cannot be decoded
    let cfg = Config { vocab: 100, ..cfg };
    let q = d.join("q.gguf");
    std::fs::write(&q, to_gguf(&Gpt2::init(cfg, 1), Quant::F32)).unwrap();
    assert!(load(q.to_str().unwrap())
        .err()
        .unwrap()
        .contains("no tokenizer"));
}

#[test]
fn mean_loss_needs_two_tokens_and_is_finite() {
    use ferrum_ref::gpt2::{Config, Gpt2};
    let m = Gpt2::init(
        Config {
            vocab: 20,
            ctx: 8,
            dim: 8,
            heads: 2,
            layers: 1,
            hidden: 16,
        },
        1,
    );
    let (l, n) = score::mean_loss(&m, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 4).unwrap();
    assert!(l.is_finite() && l > 0.0 && n > 0);
    assert!(score::mean_loss(&m, &[1], 4)
        .unwrap_err()
        .contains("at least 2"));
}
