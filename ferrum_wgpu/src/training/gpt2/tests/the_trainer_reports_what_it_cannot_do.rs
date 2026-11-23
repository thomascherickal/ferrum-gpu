use super::*;

#[test]
fn a_batch_beyond_the_budget_is_refused_by_loss_and_grads() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut plan = static_plan(&c);
    accum_plan(&c)
        .rows()
        .iter()
        .for_each(|(n, b)| plan.add(n, *b));
    activation_plan(&c, 1, 6)
        .rows()
        .iter()
        .for_each(|(n, b)| plan.add(n, *b));
    let budget = Budget {
        bytes: plan.total() + 64,
        source: Source::Override,
    };
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 61), Some(budget)).unwrap();
    let (ids, targets) = batch_data(&c, 4, 6, 0);
    let e = tr.loss_and_grads(&ids, &targets, 4).unwrap_err();
    assert!(
        e.to_string()
            .starts_with("batch 4 x 6 tokens does not fit: "),
        "{e}"
    );
}

#[test]
fn unit_numbers_beyond_32_bits_are_refused() {
    let Some(g) = open_for_test() else { return };
    let tr = GpuTrainer::new(&g, &Gpt2::init(cfg(), 62), None).unwrap();
    let e = tr.set_dropout_offset(1 << 32).unwrap_err();
    assert_eq!(e.to_string(), "dropout: unit indices exceed 32 bits");
    tr.set_dropout_offset(7).unwrap();
}

#[test]
fn a_snapshot_whose_weights_cannot_be_read_is_an_error() {
    let Some(g) = open_for_test() else { return };
    let tr = GpuTrainer::new(&g, &Gpt2::init(cfg(), 63), None).unwrap();
    tr.params[0].buf.destroy();
    let e = Engine::snapshot(&tr).unwrap_err();
    assert!(e.contains("GPU"), "{e}");
}
