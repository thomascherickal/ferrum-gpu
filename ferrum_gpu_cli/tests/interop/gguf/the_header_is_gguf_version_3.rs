use super::*;

#[test]
fn the_header_is_gguf_version_3() {
    for (label, p) in fixture().all() {
        let (b, _) = load(p);
        assert_eq!(&b[..4], b"GGUF", "{label}");
        assert_eq!(
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
            3,
            "{label}: version"
        );
    }
}

#[test]
fn tensor_data_is_aligned_ordered_and_inside_the_file() {
    for (label, p) in fixture().all() {
        let (b, g) = load(p);
        let align = 32u64;
        assert_eq!(g.data_offset() as u64 % align, 0, "{label}: data start");
        let mut end = 0u64;
        let mut tensors: Vec<_> = g.tensors.iter().collect();
        tensors.sort_by_key(|t| t.offset);
        for t in tensors {
            assert_eq!(t.offset % align, 0, "{label}: {} offset", t.name);
            assert!(
                t.offset >= end,
                "{label}: {} overlaps the one before",
                t.name
            );
            let bytes = g.tensor_bytes(&t.name).unwrap().len() as u64;
            end = t.offset + bytes;
            assert!(
                g.data_offset() as u64 + end <= b.len() as u64,
                "{label}: {} past the end",
                t.name
            );
        }
    }
}

#[test]
fn the_metadata_llama_cpp_reads_for_gpt2_is_present_and_consistent() {
    let (_, g) = load(&fixture().f32);
    assert_eq!(g.string("general.architecture").unwrap(), "gpt2");
    assert!(g.string("general.name").is_ok());
    let (ctx, dim, blocks) = (
        g.uint("gpt2.context_length").unwrap(),
        g.uint("gpt2.embedding_length").unwrap(),
        g.uint("gpt2.block_count").unwrap(),
    );
    assert_eq!((ctx, dim, blocks), (16, 32, 2));
    assert_eq!(g.uint("gpt2.feed_forward_length").unwrap(), 64);
    assert_eq!(g.uint("gpt2.attention.head_count").unwrap(), 2);
    assert!(g.get("gpt2.attention.layer_norm_epsilon").is_some());
}

#[test]
fn the_tokenizer_is_embedded_the_way_llama_cpp_reads_a_gpt2_vocabulary() {
    for (label, p) in fixture().all() {
        let (_, g) = load(p);
        assert_eq!(g.string("tokenizer.ggml.model").unwrap(), "gpt2", "{label}");
        let vocab = match g.get("tokenizer.ggml.tokens") {
            Some(ferrum_gguf::Value::Arr(a)) => a.len(),
            other => panic!("{label}: tokens {other:?}"),
        };
        assert_eq!(
            g.tensor("token_embd.weight").unwrap().dims,
            [32, vocab as u64],
            "{label}: [dim, vocab]"
        );
        assert!(g.get("tokenizer.ggml.merges").is_some(), "{label}: merges");
        assert!(
            g.uint("tokenizer.ggml.eos_token_id").is_ok(),
            "{label}: eos"
        );
    }
}
