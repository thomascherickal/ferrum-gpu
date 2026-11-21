use super::*;

#[test]
fn a_prefix_hides_the_held_out_tail() {
    let toks: Vec<usize> = (0..30).collect();
    let p = Prefix {
        inner: &toks,
        len: 20,
    };
    assert_eq!((TokenSource::len(&p), p.is_empty()), (20, false));
    assert_eq!(p.window(5, 3).unwrap(), vec![5, 6, 7]);
    assert!(p.window(18, 3).unwrap_err().contains("first 20"));
    let c = TrainConfig {
        context: 4,
        batch: 2,
        epochs: 1,
        ..TrainConfig::default()
    };
    let mut m = Mock::default();
    train(&mut m, &p, &c, None, None, &mut NoHooks).unwrap();
    assert!(m
        .calls
        .iter()
        .all(|call| call.0.iter().chain(&call.1).all(|&t| t < 20)));
    assert!(
        Prefix {
            inner: &toks,
            len: 99
        }
        .len()
            == 30,
        "never longer than the source"
    );
    assert!(TokenSource::is_empty(&Vec::<usize>::new()));
    assert!(toks.window(25, 9).unwrap_err().contains("outside"));
}
