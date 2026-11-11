use super::*;
use ferrum_gpu_gui::engine::{fresh_model, CpuBackend};
use ferrum_model::gpt2_gguf::{from_gguf, to_gguf_with, Quant};
use ferrum_ref::gpt2::Config;

fn req(max_new: usize) -> GenReq {
    GenReq {
        max_new,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 1,
        stops: vec![],
    }
}

fn gpt2() -> Gpt2Decoder {
    let tok = ByteBpeTokenizer::train_with_docs(&"abab cdcd low lower ".repeat(10), 280).unwrap();
    let cfg = Config {
        vocab: tok.vocab_size(),
        ctx: 16,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    };
    let bytes = to_gguf_with(&Gpt2::init(cfg, 2), Quant::F32, Some(&tok));
    let g = ferrum_gguf::Gguf::parse(bytes).unwrap();
    let model = from_gguf(&g).unwrap();
    Gpt2Decoder::new(Arc::new(CpuBackend::new(1)), Loaded { model, tok })
}

#[test]
fn cut_at_stop_takes_the_earliest_stop() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(cut_at_stop("hello world".into(), &s(&["wor", "lo"])), "hel");
    assert_eq!(cut_at_stop("hello".into(), &s(&["zz", ""])), "hello");
    assert_eq!(cut_at_stop("hello".into(), &[]), "hello");
}

#[test]
fn logprobs_are_normalised_and_flag_the_argmax() {
    let logits = [0.0, 10.0, 0.0, 0.0, 10.0, 0.0]; // two rows, vocab 3
    let lp = logprobs(&logits, 3, &[0, 1, 2]);
    assert_eq!(lp.len(), 2);
    assert!(lp[0].1 && !lp[1].1, "{lp:?}");
    assert!(lp.iter().all(|(l, _)| *l <= 0.0 && l.is_finite()));
    assert!(logprobs(&logits, 3, &[0]).is_empty());
}

#[test]
fn llama_decoder_scores_generates_and_refuses_gpt2_only_knobs() {
    let d = ModelDecoder::new(
        Arc::new(CpuBackend::new(1)),
        fresh_model("tiny", 1).unwrap(),
    );
    let i = d.info();
    assert_eq!(
        (i.kind.as_str(), i.arch.as_str(), i.vocab, i.ctx),
        ("cpu", "llama", 256, 64)
    );
    assert_eq!(d.encode("ab").unwrap(), vec![97, 98]);
    assert_eq!(d.bos(), 0);
    let s = d.score(&[1, 2, 3, 4]).unwrap();
    assert_eq!(s.len(), 3);
    assert!(s.iter().all(|(lp, _)| *lp <= 0.0 && lp.is_finite()));
    assert!(d.score(&[]).is_err());
    let (text, n) = d.complete("ab", &req(4)).unwrap();
    assert_eq!(n, 4);
    assert!(text.len() >= 4 || !text.is_empty());
    let mut r = req(2);
    r.top_p = 0.5;
    assert!(d
        .complete("ab", &r)
        .unwrap_err()
        .contains("only supported for GPT-2"));
}

#[test]
fn gpt2_decoder_uses_its_tokenizer() {
    let d = gpt2();
    let i = d.info();
    assert_eq!((i.arch.as_str(), i.ctx), ("gpt2", 16));
    assert_eq!(i.params, d.model.num_params());
    assert_eq!(d.bos(), d.tok.eos_id().unwrap());
    let ids = d.encode("abab cdcd").unwrap();
    assert!(
        ids.len() < "abab cdcd".len(),
        "BPE merges should shorten the text"
    );
    let (text, n) = d.complete("abab", &req(6)).unwrap();
    assert!(n > 0 && n <= 6, "{n}");
    let _ = text;
    let (cut, _) = d
        .complete(
            "abab",
            &GenReq {
                stops: vec![" ".into()],
                ..req(12)
            },
        )
        .unwrap();
    assert!(!cut.contains(' '), "{cut:?}");
    let s = d.score(&ids).unwrap();
    assert_eq!(s.len(), ids.len() - 1);
}

#[test]
fn stub_is_deterministic() {
    let d = StubDecoder;
    assert_eq!(d.complete("x", &req(3)).unwrap(), ("!!!".to_string(), 3));
    assert_eq!(
        d.complete(
            "x",
            &GenReq {
                stops: vec!["!".into()],
                ..req(3)
            }
        )
        .unwrap()
        .0,
        ""
    );
    assert_eq!(
        d.score(&[1, 2, 3]).unwrap(),
        vec![(-1.0, true), (-1.0, false)]
    );
    assert_eq!(
        (d.info().ctx, d.bos(), d.encode("a").unwrap()),
        (16, 0, vec![97])
    );
}
