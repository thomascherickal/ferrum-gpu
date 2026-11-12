use super::args;
use crate::cmds::gpt2::extras::{parse, Extras};

#[test]
fn no_flags_means_neither_option() {
    assert_eq!(
        parse(&args("")).unwrap(),
        Extras {
            tie: false,
            dropout: 0.0
        }
    );
}

#[test]
fn tie_and_dropout_are_read_independently() {
    assert_eq!(
        parse(&args("--tie")).unwrap(),
        Extras {
            tie: true,
            dropout: 0.0
        }
    );
    assert_eq!(
        parse(&args("--dropout 0.1")).unwrap(),
        Extras {
            tie: false,
            dropout: 0.1
        }
    );
    assert_eq!(
        parse(&args("--tie --dropout 0.25")).unwrap(),
        Extras {
            tie: true,
            dropout: 0.25
        }
    );
    assert_eq!(parse(&args("--dropout 0")).unwrap().dropout, 0.0);
}

#[test]
fn rates_outside_zero_to_one_are_refused_by_name() {
    for bad in ["-0.1", "1", "1.5", "nan"] {
        let e = parse(&args(&format!("--dropout {bad}"))).unwrap_err();
        assert!(e.contains("--dropout"), "{bad}: {e}");
    }
    assert!(parse(&args("--dropout x"))
        .unwrap_err()
        .contains("--dropout"));
}

#[test]
fn a_rate_above_the_maximum_is_accepted() {
    // it is used as 0.95 (and the run says so); only 1 and above are errors
    assert_eq!(parse(&args("--dropout 0.99")).unwrap().dropout, 0.99);
}
