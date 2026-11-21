use super::*;

#[test]
fn a_gguf_exported_model_runs_identically_after_a_round_trip() {
    use ferrum_gguf::Gguf;
    use ferrum_model::gpt2_gguf::{from_gguf, to_gguf, Quant};
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 7);
    let loaded = from_gguf(&Gguf::parse(to_gguf(&w, Quant::F32)).unwrap()).unwrap();
    let ids = [1usize, 5, 9];
    let (mut a, mut b) = (
        GpuGpt2::new(&g, &w, None).unwrap(),
        GpuGpt2::new(&g, &loaded, None).unwrap(),
    );
    for &t in &ids {
        assert_eq!(a.feed(t).unwrap(), b.feed(t).unwrap());
    }
}
