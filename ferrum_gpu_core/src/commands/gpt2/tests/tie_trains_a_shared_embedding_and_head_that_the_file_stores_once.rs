use super::*;

#[test]
fn tie_trains_a_shared_embedding_and_head_that_the_file_stores_once() {
    let d = dir("tie");
    let c = data(&d);
    let (plain, tied) = (d.join("plain.gguf"), d.join("tied.gguf"));
    ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 300",
        plain.display()
    ));
    let t = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 300 --tie",
        tied.display()
    ));
    assert!(!t.contains("training on the CPU engine"), "{t}");
    let has_head = |p: &std::path::Path| {
        ferrum_gguf::Gguf::parse(std::fs::read(p).unwrap())
            .unwrap()
            .tensor("output.weight")
            .is_some()
    };
    assert!(has_head(&plain), "an ordinary model stores its own head");
    assert!(!has_head(&tied), "a tied model stores the embedding only");
    assert!(
        std::fs::metadata(&tied).unwrap().len() < std::fs::metadata(&plain).unwrap().len(),
        "so the file is smaller"
    );
    // it loads and generates like any other file
    let g = ok(&format!(
        "generate --device cpu --model {} --prompt the --max-new 6",
        tied.display()
    ));
    assert!(g.starts_with("the"), "{g}");
    // device auto falls back to the CPU engine for --tie rather than failing
    ok(&format!(
        "train --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --epochs 1 --data {c} --out {} --vocab 300 --tie",
        d.join("auto.gguf").display()
    ));
}

#[test]
fn quant_accepts_both_spellings() {
    let t = [crate::commands::flags::val("--quant")];
    let q = |s: &str| quant(&Args::parse(&t, &words(s)).unwrap(), "int8");
    assert_eq!(q("--quant int4").unwrap(), Quant::Q4_0);
    assert_eq!(q("--quant q4_0").unwrap(), Quant::Q4_0);
    assert_eq!(q("--quant int8").unwrap(), Quant::Q8_0);
    assert_eq!(q("--quant f32").unwrap(), Quant::F32);
    assert_eq!(q("").unwrap(), Quant::Q8_0);
    for (s, want) in [
        ("--quant f16", Quant::F16),
        ("--quant fp16", Quant::F16),
        ("--quant fp4", Quant::Fp4),
        ("--quant mxfp4", Quant::Fp4),
        ("--quant q2", Quant::Q2K),
        ("--quant q2_k", Quant::Q2K),
    ] {
        assert_eq!(q(s).unwrap(), want, "{s}");
    }
    let e = q("--quant fp9").unwrap_err();
    assert!(
        e.contains("unknown 'fp9'") && e.contains(QUANT_NAMES),
        "{e}"
    );
}
