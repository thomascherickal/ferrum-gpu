use super::*;

#[test]
fn train_eval_generate_finetune_and_refusals_on_the_cpu() {
    let d = work("cpu");
    let data = d.join("data.txt");
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    let (data, m1, m2) = (data.to_str().unwrap(), d.join("m1.gguf"), d.join("m2.gguf"));
    let (m1, m2) = (m1.to_str().unwrap(), m2.to_str().unwrap());

    let t = ok(&[
        "train",
        "--device",
        "cpu",
        "--preset",
        "tiny",
        "--data",
        data,
        "--out",
        m1,
        "--steps",
        "60",
        "--batch",
        "4",
        "--context",
        "32",
    ]);
    let (first, last) = loss_pair(&t);
    assert!(last < first, "loss should fall: {t}");
    let e = ok(&["eval", "--model", m1, "--data", data]);
    assert!(e.contains("perplexity"), "{e}");
    let g = ok(&[
        "generate",
        "--device",
        "cpu",
        "--model",
        m1,
        "--prompt",
        "the c",
        "--max-new",
        "8",
    ]);
    assert!(g.starts_with("the c"), "{g}");
    assert_eq!(
        g,
        ok(&[
            "generate",
            "--device",
            "cpu",
            "--model",
            m1,
            "--prompt",
            "the c",
            "--max-new",
            "8"
        ]),
        "greedy is deterministic"
    );
    let f = ok(&[
        "finetune-gguf",
        "--device",
        "cpu",
        "--model",
        m1,
        "--data",
        data,
        "--out",
        m2,
        "--steps",
        "10",
        "--batch",
        "4",
        "--context",
        "32",
        "--quant",
        "q8_0",
    ]);
    assert!(f.contains("wrote"), "{f}");
    assert!(ok(&["info", "--device", "cpu", "--model", m2]).contains("model: gpt2 vocab"));

    // refusals exit non-zero with a message on stderr
    for bad in [
        vec!["train", "--bogus"],
        vec!["nonsense"],
        vec![
            "generate", "--device", "cpu", "--model", "/nope", "--prompt", "x",
        ],
    ] {
        let o = bin(&bad);
        assert!(!o.status.success() && !o.stderr.is_empty(), "{bad:?}");
    }
    std::fs::remove_dir_all(d).ok();
}
