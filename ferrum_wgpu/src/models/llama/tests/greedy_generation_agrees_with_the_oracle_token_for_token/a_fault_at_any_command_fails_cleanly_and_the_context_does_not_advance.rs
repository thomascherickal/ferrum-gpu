use super::*;

#[test]
fn a_fault_at_any_command_fails_cleanly_and_the_context_does_not_advance() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 13, false, false);
    let mut m = GpuLlama::new(&g, &w, None, None).unwrap();
    let commands = g.sweep_faults(|| {
        let before = m.pos();
        let r = m.feed(3).map(|_| ());
        if r.is_err() {
            assert_eq!(
                m.pos(),
                before,
                "a failed token must not advance the position"
            );
        }
        r
    });
    // 2 (index copy + gather) + 18 per block x 2 blocks + 2 (final norm + head).
    assert_eq!(commands, 40);
    assert_eq!(m.pos(), 1, "only the successful attempt advanced");
    let n = g.sweep_faults(|| m.forward_all(&[1, 2, 3]).map(|_| ()));
    // 1 gather + 17 per block x 2 (no cache copies, a 3-dispatch attention) + final norm + head.
    assert_eq!(n, 37);
    // score = the batched pass over all but the last token (1 + 17 x 2 + 2) + the score kernel.
    assert_eq!(
        g.sweep_faults(|| m.score(&[1, 2, 3, 4]).map(|_| ())),
        37 + 1
    );
    // the quantised path with biases records extra bias dispatches
    let wb = Llama::random(cfg(), 14, true, false);
    let mut mq = GpuLlama::new(&g, &wb, Some(QKind::Q8_0), None).unwrap();
    let nq = g.sweep_faults(|| mq.feed(3).map(|_| ()));
    assert_eq!(
        nq,
        40 + 3 * 2,
        "one extra add_bias per biased q/k/v per block"
    );
}
