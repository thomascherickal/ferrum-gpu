use super::args;
use crate::cmds::gpt2::extras::{parse, Extras};

#[test]
fn no_flags_means_neither_option() {
    let x = parse(&args("")).unwrap();
    assert_eq!(
        x,
        Extras {
            tie: false,
            dropout: 0.0
        }
    );
    assert!(!x.needs_cpu());
}

#[test]
fn tie_alone_does_not_need_the_cpu() {
    let x = parse(&args("--tie")).unwrap();
    assert_eq!(
        x,
        Extras {
            tie: true,
            dropout: 0.0
        }
    );
    assert!(!x.needs_cpu(), "a tied head trains on either engine");
}

#[test]
fn a_positive_dropout_needs_the_cpu_and_zero_does_not() {
    assert!(parse(&args("--dropout 0.1")).unwrap().needs_cpu());
    assert!(!parse(&args("--dropout 0")).unwrap().needs_cpu());
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
