use super::*;

#[test]
fn dropout_training_follows_the_cpu_trajectory() {
    check("dropout 0.3", compare(Mode::default(), 0.3, 12));
}

#[test]
fn a_high_rate_follows_it_too() {
    check("dropout 0.9", compare(Mode::default(), 0.9, 8));
}

#[test]
fn dropout_with_qat_follows_the_cpu_trajectory() {
    check(
        "dropout+qat",
        compare(
            Mode {
                qat: true,
                ..Mode::default()
            },
            0.2,
            10,
        ),
    );
}

#[test]
fn dropout_with_a_tied_head_follows_the_cpu_trajectory() {
    check(
        "dropout+tie",
        compare(
            Mode {
                tie: true,
                ..Mode::default()
            },
            0.2,
            10,
        ),
    );
}

#[test]
fn micro_batched_dropout_drops_the_same_units_as_the_whole_batch() {
    // the units of micro-batch c are numbered from the start of the batch, so the split run is
    // the CPU's whole-batch run up to summation order
    check(
        "dropout, 2 micro-batches",
        compare(
            Mode {
                micro: Some(2),
                ..Mode::default()
            },
            0.3,
            8,
        ),
    );
    check(
        "dropout, 4 micro-batches",
        compare(
            Mode {
                micro: Some(1),
                ..Mode::default()
            },
            0.3,
            6,
        ),
    );
}
