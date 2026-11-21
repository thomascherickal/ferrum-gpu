use super::*;

#[test]
fn untying_stops_re_deriving_the_head() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    tr.set_tie(true).unwrap();
    tr.set_tie(false).unwrap();
    assert!(!tr.tie_enabled());
    let (ids, targets) = data(&c, 2, 5, 0);
    tr.step(&ids, &targets, 2, &sc()).unwrap();
    assert!(
        !is_tied(&tr.weights_host().unwrap()),
        "the head trains on its own again"
    );
}
