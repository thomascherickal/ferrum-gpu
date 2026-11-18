use super::*;

#[test]
fn every_spelling_parses_and_every_tag_names_its_precision() {
    for (s, q) in [
        ("f32", Quant::F32),
        ("f16", Quant::F16),
        ("fp16", Quant::F16),
        ("q8_0", Quant::Q8_0),
        ("int8", Quant::Q8_0),
        ("q4_0", Quant::Q4_0),
        ("int4", Quant::Q4_0),
        ("fp4", Quant::Fp4),
        ("mxfp4", Quant::Fp4),
        ("q2", Quant::Q2K),
        ("q2_k", Quant::Q2K),
    ] {
        assert_eq!(Quant::parse(s), Some(q), "{s}");
    }
    assert_eq!(Quant::parse("fp8"), None);
    assert_eq!(Quant::parse(""), None);
    let tags: Vec<_> = [
        Quant::F32,
        Quant::F16,
        Quant::Q8_0,
        Quant::Q4_0,
        Quant::Fp4,
        Quant::Q2K,
    ]
    .map(Quant::tag)
    .to_vec();
    assert_eq!(tags, ["f32", "f16", "int8", "int4", "fp4", "q2"]);
}

#[test]
fn the_tensor_type_follows_the_block_size_of_the_precision() {
    // widths: 100 (no whole block), 32, 64, 256
    for (q, want) in [
        (Quant::F32, [GGML_F32; 4]),
        (Quant::F16, [GGML_F16; 4]),
        (Quant::Q8_0, [GGML_F32, GGML_Q8_0, GGML_Q8_0, GGML_Q8_0]),
        (Quant::Q4_0, [GGML_F32, GGML_Q4_0, GGML_Q4_0, GGML_Q4_0]),
        (Quant::Fp4, [GGML_F32, GGML_MXFP4, GGML_MXFP4, GGML_MXFP4]),
        // Q2_K needs 256; narrower multiples of 32 fall back to Q4_0, as llama.cpp does
        (Quant::Q2K, [GGML_F32, GGML_Q4_0, GGML_Q4_0, GGML_Q2_K]),
    ] {
        let got = [100, 32, 64, 256].map(|ne0| q.tensor_type(ne0));
        assert_eq!(got, want, "{q:?}");
    }
}
